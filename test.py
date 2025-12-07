"""
Mobile GUI Agent Sequential Evaluator with Latency Tracking
"""

import base64
import json
import re
import time
import random
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml
from datetime import datetime
from utils import load_and_resize_image


@dataclass
class StepRecord:
    """Ground Truth 步骤记录"""
    state: str
    choice: int
    input_text: str
    state_strs: List[str]


@dataclass
class TaskProfile:
    """任务配置"""
    task_name: str
    app_name: str
    records: List[StepRecord]

    @classmethod
    def from_yaml(cls, yaml_path: Path):
        """从 YAML 加载任务"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 获取任务名
        task_name = data.get('task_name', '') or data.get('task', '') or yaml_path.stem

        # 获取 records
        records_data = data.get('records', [])
        if not records_data:
            return None

        records = []
        for r in records_data:
            state = r.get('State', '')
            choice = r.get('Choice', -1)
            input_text = r.get('Input', 'null')
            state_str_raw = r.get('state_str', [])
            if isinstance(state_str_raw, str):
                state_strs = [state_str_raw] if state_str_raw else []
            elif isinstance(state_str_raw, list):
                state_strs = [s for s in state_str_raw if s]
            else:
                state_strs = []
            records.append(StepRecord(state, choice, input_text, state_strs))

        if not records:
            return None

        return cls(task_name, yaml_path.parent.name, records)


@dataclass
class StepResult:
    """单步评估结果"""
    step_idx: int
    gt_action: str
    gt_id: int
    gt_input: str
    pred_action: str
    pred_id: int
    pred_input: str
    action_correct: bool
    element_correct: bool
    input_correct: bool
    step_correct: bool
    prompt: str
    raw_output: str
    latency: float  # 新增：步骤延迟（秒）


@dataclass
class TaskResult:
    """任务评估结果"""
    task_name: str
    app_name: str
    total_steps: int
    step_results: List[StepResult]
    action_correct_count: int
    element_correct_count: int
    input_correct_count: int
    step_correct_count: int
    task_success: bool
    total_latency: float  # 新增：任务总延迟（秒）


class PromptBuilder:
    """构造 prompt"""

    def build_prompt(self, task_name: str, app_name: str, state: str, history: str) -> str:
        """构造完整 prompt"""
        return f"""You are a mobile GUI agent. Analyze the UI and determine the next action.

# Task
{task_name}

# Action History
{history}

# Current UI State
{state}

# Instructions
Respond in JSON format:
{{
  "finished": "yes/no",
  "id": <element_id>,
  "action": "tap/input",
  "input_text": "<text or N/A>"
}}

Rules:
- If task is complete: {{"finished": "yes", "id": -1}}
- For tap: {{"finished": "no", "id": <num>, "action": "tap", "input_text": "N/A"}}
- For input: {{"finished": "no", "id": <num>, "action": "input", "input_text": "<text>"}}

Your JSON response:"""

    def build_history(self, step_results: List[StepResult], app_name: str) -> str:
        """构造历史动作"""
        if not step_results:
            return f"- Start {app_name} app"

        history = [f"- Start {app_name} app"]
        for s in step_results:
            if s.gt_id == -1:
                continue
            if s.gt_input != 'null':
                history.append(f"- Input: element id={s.gt_id}, text='{s.gt_input}'")
            else:
                history.append(f"- Tap: element id={s.gt_id}")

        return "\n".join(history)


class OutputParser:
    """解析 LLM 输出"""

    @staticmethod
    def parse(output: str) -> Dict[str, Any]:
        """解析 LLM 返回的 JSON"""
        try:
            output = output.strip()

            # 移除 markdown 代码块
            if '```' in output:
                output = re.sub(r'^```(?:json)?\s*\n?', '', output, flags=re.MULTILINE)
                output = re.sub(r'\n?```\s*$', '', output, flags=re.MULTILINE)

            # 归一化 action 行，移除多余字符
            if '"action"' in output:
                lines = output.split('\n')
                norm_lines = []
                for line in lines:
                    if '"action"' in line:
                        m = re.search(r'"action"\s*:\s*"([^"]*)"', line)
                        if m:
                            action_val = m.group(1)
                            line = f'  "action": "{action_val}",'
                    norm_lines.append(line)
                output = '\n'.join(norm_lines)

            # 修复重复的 action 字段
            if output.count('"action"') > 1:
                lines = output.split('\n')
                new_lines = []
                action_seen = False
                for line in lines:
                    if '"action"' in line and not action_seen:
                        if line.count('"') % 2 != 0:
                            line = re.sub(r'"action"\s*:\s*"[^"]*$', '"action": "tap"', line)
                        new_lines.append(line)
                        action_seen = True
                    elif '"action"' not in line:
                        new_lines.append(line)
                output = '\n'.join(new_lines)

            # 去掉重复的逗号导致的非法 JSON
            while ',,' in output:
                output = output.replace(',,', ',')

            # 修复未闭合的 action 字段
            if '"action"' in output:
                action_match = re.search(r'"action"\s*:\s*"([^"]*?)(?:"|,|\n|$)', output, re.DOTALL)
                if action_match:
                    action_val = action_match.group(1)
                    if 'click(' in action_val or 'tap(' in action_val or '<element' in action_val:
                        output = re.sub(
                            r'"action"\s*:\s*"[^"]*?(?:click|tap|<element).*?(?:"|,|\n)',
                            '"action": "tap",',
                            output,
                            flags=re.DOTALL
                        )

            # 确保 JSON 完整
            if not output.endswith('}'):
                output = output.rstrip(',\n ') + '\n}'

            # 解析 JSON
            data = json.loads(output)

            # 检查是否完成
            finished = str(data.get('finished', '')).lower() in ['yes', 'y', 'true', '1']
            if finished:
                return {'action': 'end', 'id': -1, 'input': 'N/A'}

            # 解析字段
            elem_id = data.get('id', -1)
            if isinstance(elem_id, str):
                if elem_id.lower() in ['n/a', 'none', '']:
                    elem_id = -1
                else:
                    try:
                        elem_id = int(elem_id)
                    except:
                        elem_id = -1

            action = str(data.get('action', 'tap')).lower()
            if any(k in action for k in ['tap', 'click', 'check', 'select']):
                action = 'tap'
            elif any(k in action for k in ['input', 'type', 'enter']):
                action = 'input'
            elif elem_id == -1:
                action = 'end'

            input_text = str(data.get('input_text', 'N/A'))
            if input_text.lower() in ['n/a', 'none', 'null', '']:
                input_text = 'N/A'

            return {'action': action, 'id': int(elem_id), 'input': input_text}

        except Exception as e:
            print(f"[PARSE ERROR] {e}")
            print(f"Output: {output[:200]}")
            return {'action': 'end', 'id': -1, 'input': 'N/A'}


class Evaluator:
    """评估器主类"""

    def __init__(
        self,
        dataset_root: str,
        output_dir: str = "./eval_results",
        verbose: bool = True,
        llm_api_url: str = "http://localhost:8080/v1/chat/completions",
        llm_temperature: float = 0.0,
        llm_max_tokens: int = 300,
        llm_retry_times: int = 12,
        specify_apps: Optional[List[str]] = None,
        use_images: bool = True
    ):
        """初始化评估器"""
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.specify_apps = specify_apps
        self.use_images = use_images

        self.llm_api_url = llm_api_url
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.llm_retry_times = llm_retry_times

        self.prompt_builder = PromptBuilder()
        self.parser = OutputParser()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_log_file = self.output_dir / "sent_screenshots.log"
        self.screenshot_log_file.touch(exist_ok=True)
        self.all_task_results: List[TaskResult] = []
        self.state_image_cache: Dict[str, Dict[str, Path]] = {}
        self._disable_images_globally = False

    def load_dataset(self) -> Dict[str, List[Path]]:
        """加载数据集"""
        dataset = {}
        for app_dir in sorted(self.dataset_root.iterdir()):
            if not app_dir.is_dir():
                continue

            app_name = app_dir.name
            if self.specify_apps is not None and app_name not in self.specify_apps:
                continue

            yaml_files = sorted(list(app_dir.glob("*.yaml")))
            if yaml_files:
                dataset[app_name] = yaml_files

        return dataset

    def build_state_image_map(self, app_name: str) -> Dict[str, Path]:
        """为某个 app 构建 state_str -> screenshot 映射"""
        if app_name in self.state_image_cache:
            return self.state_image_cache[app_name]

        mapping: Dict[str, Path] = {}
        events_dir = self.dataset_root / app_name / "events"
        states_dir = self.dataset_root / app_name / "states"

        # 1) 优先尝试事件同名 PNG
        if events_dir.exists():
            for event_file in sorted(events_dir.glob("*.json")):
                png_path = event_file.with_suffix(".png")
                if not png_path.exists():
                    continue
                try:
                    with open(event_file, 'r') as f:
                        event_data = json.load(f)
                    for key in ["state_str", "start_state", "stop_state"]:
                        state_hash = event_data.get(key)
                        if isinstance(state_hash, str) and state_hash:
                            mapping[state_hash] = png_path
                        elif isinstance(state_hash, list):
                            for h in state_hash:
                                if h:
                                    mapping[h] = png_path
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed to load event {event_file.name}: {e}")
                    continue

        # 2) 再尝试 states 目录：state_* 配 screen_* (同时间戳)
        if states_dir.exists():
            for state_file in sorted(states_dir.glob("state_*.json")):
                tag = state_file.stem.replace("state_", "")
                png_path = states_dir / f"screen_{tag}.png"
                if not png_path.exists():
                    continue
                try:
                    with open(state_file, 'r') as f:
                        state_data = json.load(f)
                    state_hash = state_data.get("state_str")
                    if isinstance(state_hash, str) and state_hash:
                        mapping[state_hash] = png_path
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Failed to load state {state_file.name}: {e}")
                    continue

        self.state_image_cache[app_name] = mapping
        return mapping

    @staticmethod
    def find_screenshot(state_hashes: List[str], state_image_map: Dict[str, Path]) -> Optional[Path]:
        """根据 state_str 查找对应截图"""
        for h in state_hashes:
            if h in state_image_map:
                return state_image_map[h]
        return None

    @staticmethod
    def debug_missing_screenshot(app_name: str, state_hashes: List[str], state_image_map: Dict[str, Path]):
        missing = [h for h in state_hashes if h not in state_image_map]
        if missing:
            print(f"[DEBUG] No screenshot for app={app_name}, hashes={missing}")
        return None

    def log_screenshot(self, image_path: Path, identifier: str):
        """记录发送的截图文件名"""
        try:
            with open(self.screenshot_log_file, "a", encoding="utf-8") as f:
                f.write(f"{identifier}: {image_path.name}\n")
            if self.verbose:
                print(f"[DEBUG] Screenshot logged: {image_path.name}")
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Failed to log screenshot {image_path}: {e}")

    def query_llm(self, prompt: str, identifier: str = "", image_path: Optional[Path] = None) -> str:
        """
        使用与你提供的 inference_chat_llama_cpp 相同的请求格式，
        能够稳定发送 screenshot 到 llama.cpp。
        """

        import requests

        # 构造 messages
        content_items = [{"type": "text", "text": prompt}]

        # 如果有图片，则压缩后附加 image_url（data uri）
        if image_path is not None and image_path.exists():
            try:
                b64 = load_and_resize_image(image_path)  # <-- 我们前面写好的压缩函数
                data_uri = f"data:image/jpeg;base64,{b64}"

                content_items.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })

                # 日志
                self.log_screenshot(image_path, identifier)

            except Exception as e:
                if self.verbose:
                    print(f"[WARN] screenshot encode failed for {identifier}: {e}")

        # openai-compatible chat format
        messages = [{
            "role": "user",
            "content": content_items
        }]

        payload = {
            "messages": messages,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "stream": False
        }

        # 发送请求
        try:
            res = requests.post(self.llm_api_url, json=payload, timeout=900)
            if res.status_code != 200:
                print(f"[Error] LLM Response: {res.text}")
            res.raise_for_status()

            return res.json()["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"[ERROR] {identifier}: {e}")
            return json.dumps({"finished": "yes", "id": -1})


    def evaluate_step(
        self,
        step_idx: int,
        gt_record: StepRecord,
        llm_output: Dict[str, Any],
        prompt: str,
        raw_output: str,
        latency: float
    ) -> StepResult:
        """评估单个步骤"""
        # 确定 GT 动作类型
        if gt_record.choice == -1:
            gt_action = 'end'
        elif gt_record.input_text != 'null':
            gt_action = 'input'
        else:
            gt_action = 'tap'

        # 判断正确性
        element_correct = (llm_output['id'] == gt_record.choice)
        action_correct = (llm_output['action'] == gt_action)

        # 判断输入
        if gt_record.input_text != 'null':
            input_correct = (
                llm_output['action'] == 'input' and
                llm_output['id'] == gt_record.choice and
                llm_output['input'] == gt_record.input_text
            )
        else:
            input_correct = True

        step_correct = element_correct and action_correct and input_correct

        return StepResult(
            step_idx=step_idx,
            gt_action=gt_action,
            gt_id=gt_record.choice,
            gt_input=gt_record.input_text,
            pred_action=llm_output['action'],
            pred_id=llm_output['id'],
            pred_input=llm_output['input'],
            action_correct=action_correct,
            element_correct=element_correct,
            input_correct=input_correct,
            step_correct=step_correct,
            prompt=prompt,
            raw_output=raw_output,
            latency=latency
        )

    def run_task(self, task_profile: TaskProfile) -> TaskResult:
        """运行单个任务"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {task_profile.task_name} ({task_profile.app_name})")
            print(f"Steps: {len(task_profile.records)}")
            print(f"{'='*60}")

        step_results = []
        state_image_map = self.build_state_image_map(task_profile.app_name)
        task_start_time = time.time()

        for step_idx, gt_record in enumerate(task_profile.records):
            if self.verbose:
                print(f"\n--- Step {step_idx + 1}/{len(task_profile.records)} ---")

            # 构造 history
            history = self.prompt_builder.build_history(step_results, task_profile.app_name)

            # 构造 prompt
            prompt = self.prompt_builder.build_prompt(
                task_name=task_profile.task_name,
                app_name=task_profile.app_name,
                state=gt_record.state,
                history=history
            )

            if self.verbose:
                gt_action = 'end' if gt_record.choice == -1 else ('input' if gt_record.input_text != 'null' else 'tap')
                print(f"GT:   {gt_action}, id={gt_record.choice}, input={gt_record.input_text}")

            # 调用 LLM 并计时
            identifier = f"{task_profile.app_name}/{task_profile.task_name}/step_{step_idx}"
            image_path = self.find_screenshot(gt_record.state_strs, state_image_map)
            if self.verbose and image_path:
                print(f"Screenshot: {image_path}")
            elif self.verbose:
                self.debug_missing_screenshot(task_profile.app_name, gt_record.state_strs, state_image_map)
                print("Screenshot: None found for this step")
            
            step_start_time = time.time()
            raw_output = self.query_llm(prompt, identifier, image_path=image_path)
            step_latency = time.time() - step_start_time

            # 解析输出
            llm_output = self.parser.parse(raw_output)

            if self.verbose:
                print(f"Pred: {llm_output['action']}, id={llm_output['id']}, input={llm_output['input']}")
                print(f"Latency: {step_latency:.2f}s")

            # 评估
            step_result = self.evaluate_step(step_idx, gt_record, llm_output, prompt, raw_output, step_latency)
            step_results.append(step_result)

            if self.verbose:
                status = "✓" if step_result.step_correct else "✗"
                print(f"Result: {status}")

        task_total_latency = time.time() - task_start_time

        # 计算任务级别指标
        action_correct = sum(1 for r in step_results if r.action_correct)
        element_correct = sum(1 for r in step_results if r.element_correct)
        input_correct = sum(1 for r in step_results if r.input_correct)
        step_correct = sum(1 for r in step_results if r.step_correct)
        task_success = (step_correct == len(step_results))

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Task: {'SUCCESS ✓' if task_success else 'FAILED ✗'}")
            print(f"Steps: {step_correct}/{len(step_results)}")
            print(f"Total Latency: {task_total_latency:.2f}s")
            print(f"Average Step Latency: {task_total_latency/len(step_results):.2f}s")
            print(f"{'='*60}")

        return TaskResult(
            task_name=task_profile.task_name,
            app_name=task_profile.app_name,
            total_steps=len(step_results),
            step_results=step_results,
            action_correct_count=action_correct,
            element_correct_count=element_correct,
            input_correct_count=input_correct,
            step_correct_count=step_correct,
            task_success=task_success,
            total_latency=task_total_latency
        )

    def run_evaluation(self) -> Dict[str, Any]:
        """运行完整评估"""
        dataset = self.load_dataset()

        print(f"\n{'#'*60}")
        print(f"# Starting Evaluation")
        if self.specify_apps:
            print(f"# Apps: {', '.join(self.specify_apps)}")
        print(f"# Total: {len(dataset)} apps, {sum(len(t) for t in dataset.values())} tasks")
        print(f"{'#'*60}")

        self.all_task_results = []

        for app_name, yaml_files in sorted(dataset.items()):
            print(f"\n{'#'*60}")
            print(f"# App: {app_name} ({len(yaml_files)} tasks)")
            print(f"{'#'*60}")

            for yaml_path in yaml_files:
                task = TaskProfile.from_yaml(yaml_path)

                if task is None:
                    if self.verbose:
                        print(f"[SKIP] {yaml_path.name} - no valid records")
                    continue

                if len(task.records) == 0:
                    if self.verbose:
                        print(f"[SKIP] {task.task_name} - 0 steps")
                    continue

                result = self.run_task(task)
                self.all_task_results.append(result)

        metrics = self.calculate_metrics()
        self.save_results(metrics)

        return metrics

    def calculate_metrics(self) -> Dict[str, Any]:
        """计算总体指标"""
        if not self.all_task_results:
            return {}

        total_tasks = len(self.all_task_results)
        total_steps = sum(r.total_steps for r in self.all_task_results)

        action_correct = sum(r.action_correct_count for r in self.all_task_results)
        element_correct = sum(r.element_correct_count for r in self.all_task_results)
        step_correct = sum(r.step_correct_count for r in self.all_task_results)
        task_success = sum(1 for r in self.all_task_results if r.task_success)

        # 输入准确率
        input_total = 0
        input_correct = 0
        for r in self.all_task_results:
            for s in r.step_results:
                if s.gt_input != 'null':
                    input_total += 1
                    if s.input_correct:
                        input_correct += 1

        # 延迟统计
        total_latency = sum(r.total_latency for r in self.all_task_results)
        avg_task_latency = total_latency / total_tasks if total_tasks > 0 else 0
        
        all_step_latencies = []
        for r in self.all_task_results:
            for s in r.step_results:
                all_step_latencies.append(s.latency)
        avg_step_latency = sum(all_step_latencies) / len(all_step_latencies) if all_step_latencies else 0

        metrics = {
            'overall': {
                'action_accuracy': round(action_correct / total_steps, 4) if total_steps > 0 else 0,
                'element_accuracy': round(element_correct / total_steps, 4) if total_steps > 0 else 0,
                'input_accuracy': round(input_correct / input_total, 4) if input_total > 0 else 1.0,
                'step_accuracy': round(step_correct / total_steps, 4) if total_steps > 0 else 0,
                'task_success_rate': round(task_success / total_tasks, 4) if total_tasks > 0 else 0,
                'total_tasks': total_tasks,
                'successful_tasks': task_success,
                'total_steps': total_steps,
                'correct_steps': step_correct,
                'total_latency': round(total_latency, 2),
                'avg_task_latency': round(avg_task_latency, 2),
                'avg_step_latency': round(avg_step_latency, 2)
            }
        }

        return metrics

    def save_results(self, metrics: Dict[str, Any]):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果
        detailed_results = []
        for task_result in self.all_task_results:
            detailed_results.append({
                'task_name': task_result.task_name,
                'app_name': task_result.app_name,
                'total_steps': task_result.total_steps,
                'task_success': task_result.task_success,
                'total_latency': round(task_result.total_latency, 2),
                'avg_step_latency': round(task_result.total_latency / task_result.total_steps, 2) if task_result.total_steps > 0 else 0,
                'steps': [
                    {
                        'step': s.step_idx,
                        'gt': {'action': s.gt_action, 'id': s.gt_id, 'input': s.gt_input},
                        'pred': {'action': s.pred_action, 'id': s.pred_id, 'input': s.pred_input},
                        'correct': s.step_correct,
                        'latency': round(s.latency, 2)
                    }
                    for s in task_result.step_results
                ]
            })

        with open(self.output_dir / f'results_{timestamp}.json', 'w') as f:
            json.dump({'metrics': metrics, 'details': detailed_results}, f, indent=2)

        # 打印摘要
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Action Accuracy:    {metrics['overall']['action_accuracy']:.2%}")
        print(f"Element Accuracy:   {metrics['overall']['element_accuracy']:.2%}")
        print(f"Input Accuracy:     {metrics['overall']['input_accuracy']:.2%}")
        print(f"Step Accuracy:      {metrics['overall']['step_accuracy']:.2%}")
        print(f"Task Success Rate:  {metrics['overall']['task_success_rate']:.2%}")
        print(f"\nSteps:  {metrics['overall']['correct_steps']}/{metrics['overall']['total_steps']}")
        print(f"Tasks:  {metrics['overall']['successful_tasks']}/{metrics['overall']['total_tasks']}")
        print(f"\n--- Latency Statistics ---")
        print(f"Total Latency:           {metrics['overall']['total_latency']:.2f}s")
        print(f"Avg Task Latency:        {metrics['overall']['avg_task_latency']:.2f}s")
        print(f"Avg Step Latency:        {metrics['overall']['avg_step_latency']:.2f}s")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # 评估所有 app
    evaluator = Evaluator(
        dataset_root="./data/user_tasks",
        output_dir="./eval_results",
        specify_apps=["clock"],  # None = 全部，或 ["navigation", "raw_utgs"]
        verbose=True,
        use_images=False  # llama.cpp 默认关闭图片，若支持多模态可改为 True
    )

    metrics = evaluator.run_evaluation()