import json
import logging
import re
from copy import deepcopy

import torch

from rllm.tools.tool_base import Tool, ToolCall, ToolOutput

from .utils import PARSER_TEST_MESSAGES

logger = logging.getLogger(__name__)


class ChatTemplateParser:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.generation_prompt = self._get_generation_prompt(tokenizer)

    def _get_generation_prompt(self, tokenizer):
        messages = [{"role": "assistant", "content": ""}]

        with_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        without_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)

        generation_prompt = with_prompt[len(without_prompt) :]

        return generation_prompt

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        if self.processor is not None:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        else:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def parse_completion(self, completion_ids: list[int]):
        raise NotImplementedError("ChatTemplateParser does not support parse_completion")

    def verify_equivalence(self, messages, verbose=True):
        """Verify that parsing messages together is equivalent to parsing them individually.

        Args:
            messages (list): List of message dictionaries to test
            verbose (bool): Whether to print detailed information about the test

        Returns:
            bool: True if the equivalence check passes, False otherwise

        Raises:
            AssertionError: If the equivalence check fails and verbose is True
        """
        # Parse all messages together
        batch_result = self.parse(messages)

        # Parse each message individually and concatenate
        individual_results = []
        for message in messages:
            individual_results.append(self.parse([message]))

        concatenated_result = "".join(individual_results)

        # Check if results are equivalent
        is_equivalent = batch_result == concatenated_result

        if verbose and not is_equivalent:
            print("Equivalence check failed!")
            print("Batch parsing result:")
            print(batch_result)
            print("\nConcatenated individual parsing result:")
            print(concatenated_result)
            raise AssertionError("Parser failed equivalence check. See above for details.")

        return is_equivalent

    @classmethod
    def get_parser(cls, tokenizer, processor=None, disable_thinking=False) -> "ChatTemplateParser":
        """Factory method to get the appropriate parser based on a string identifier.

        Args:
            parser_type (str): String identifier for the parser type
            tokenizer: The tokenizer to use with the parser
            disable_thinking: Whether generation prompt will disable thinking.

        Returns:
            ChatTemplateParser: An instance of the requested parser

        Raises:
            ValueError: If the parser_type is not recognized
        """
        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            logger.info(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                logger.info(f"Using DeepseekQwenChatTemplateParser for {tokenizer.name_or_path}")
                return DeepseekQwenChatTemplateParser(tokenizer)
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                logger.info(f"Using QwenChatTemplateParser for {tokenizer.name_or_path}")
                return QwenChatTemplateParser(tokenizer, processor=processor, disable_thinking=disable_thinking)
            elif "llama" in model_name:
                logger.info(f"Using LlamaChatTemplateParser for {tokenizer.name_or_path}")
                return LlamaChatTemplateParser(tokenizer)

        # Default to the standard parser if no specific match
        parser = ChatTemplateParser(tokenizer, processor=processor)
        logger.info(f"No custom parser found. Using default ChatTemplateParser for {tokenizer.name_or_path}")
        assert parser.verify_equivalence(PARSER_TEST_MESSAGES), "Parser failed equivalence check"
        return parser

    def tokenize_and_mask(self, messages):
        try:
            last_assistant_idx = max(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except ValueError:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:last_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        response = self.parse([messages[last_assistant_idx]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
        response = response[len(self.generation_prompt) :].rstrip("\n")  # handle qwen trailing newline from eot token
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)
        response_mask = [1] * len(response_ids)

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask

    def tokenize_and_mask_cumulative(self, messages):
        response_ids = []
        response_mask = []

        try:
            first_assistant_idx = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        prompt = self.parse(messages[:first_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        for i in range(first_assistant_idx, len(messages)):
            is_asst = messages[i]["role"] == "assistant"
            if is_asst:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
                response = response[len(self.generation_prompt) :]
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([1] * len(ids))
            else:
                response = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=True, accumulate_reasoning=False)
                ids = self.tokenizer.encode(response, add_special_tokens=False)
                response_ids.extend(ids)
                response_mask.extend([0] * len(ids))

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        return prompt_ids, response_ids, response_mask

    def tokenize_and_mask_cumulative_mm(self, messages, max_prompt_len=None, max_response_len=None):
        """Multimodal variant of tokenize_and_mask_cumulative.

        Base implementation is text-only: returns the same ids/mask as
        tokenize_and_mask_cumulative plus an empty multi_modal_inputs dict, so
        callers can use a single code path. Parsers that support images
        (QwenChatTemplateParser) override this to expand <|image_pad|>
        placeholders into real patch tokens and return pixel_values /
        image_grid_thw. max_prompt_len / max_response_len are ignored here.
        """
        prompt_ids, response_ids, response_mask = self.tokenize_and_mask_cumulative(messages)
        return prompt_ids, response_ids, response_mask, {}


class DeepseekQwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.system_token = ""
        self.user_token = "<｜User｜>"
        self.assistant_token = "<｜Assistant｜>"
        self.generation_prompt = self.assistant_token + "<think>\n"

        from rllm.parser.tool_parser import R1ToolParser

        self.tool_parser = R1ToolParser()

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool | dict] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                import traceback

                traceback.print_exc()
                logger.error(f"Failed to format tools: {e}")

        result = ""

        if is_first_msg:
            result += self.bos_token

        if is_first_msg and messages[0]["role"] != "system" and tools_prompt_str:
            result += self.system_token + tools_prompt_str

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content

    def parse_user(self, message):
        return self.user_token + message["content"]

    def parse_assistant(self, message, accumulate_reasoning=False):
        content = (message.get("content", None) or "").strip()
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not accumulate_reasoning:
            return self.assistant_token + content + self.eos_token
        elif not reasoning:
            return self.assistant_token + "<think>\n" + content + self.eos_token
        else:
            result = self.assistant_token

            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning + "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        # Avoid mutating original message structures by parsing into a local variable
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_json = f"```json\n{json.dumps(arguments_obj)}\n```"
                        tool_call_str = f"{self.tool_parser.tool_call_begin}function{self.tool_parser.tool_sep}{tool_call_dict['name']}\n{tool_call_json}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    joined_calls_str = "\n".join(tool_calls_strs)
                    tool_calls_str = f"{self.tool_parser.tool_calls_begin}\n{joined_calls_str}\n{self.tool_parser.tool_calls_end}"
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eos_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            reasoning = None
            content = completion_text
            if content.startswith("<think>"):
                content = content[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            content = content.strip()

        tool_calls = self.tool_parser.parse(completion_text)

        begin_pattern = re.escape(self.tool_parser.tool_call_begin)
        end_pattern = re.escape(self.tool_parser.tool_call_end)
        content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)

        wrapper_begin_pattern = re.escape(self.tool_parser.tool_calls_begin)
        wrapper_end_pattern = re.escape(self.tool_parser.tool_calls_end)
        content = re.sub(f"{wrapper_begin_pattern}.*?{wrapper_end_pattern}", "", content, flags=re.DOTALL)

        content = content.strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }


class QwenChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer, processor=None, disable_thinking=False):
        super().__init__(tokenizer, processor=processor)
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.eot_token = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\n\n</think>\n\n"
        self.generation_prompt = self.assistant_token
        self.image_token = "<|image_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

        from rllm.parser.tool_parser import QwenToolParser

        self.tool_parser = QwenToolParser()

    def parse(self, messages: list[dict], add_generation_prompt: bool = False, is_first_msg: bool = False, tools: list[Tool] = None, accumulate_reasoning: bool = False, **kwargs) -> str:
        tools = tools or []
        tools_prompt_str = ""
        if tools:
            try:
                tool_schema_strs = []
                for tool in tools:
                    if isinstance(tool, Tool):
                        tool_schema_str = json.dumps(tool.json)
                    elif isinstance(tool, dict):
                        tool_schema_str = json.dumps(tool)
                    else:
                        tool_schema_str = tool
                    tool_schema_strs.append(tool_schema_str)
                tools_schema_str = "\n".join(tool_schema_strs)
                tools_prompt_str = self.tool_parser.get_tool_prompt(tools_schema_str)
            except Exception as e:
                logger.error(f"Failed to format tools: {e}")

        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += self.system_token + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." + tools_prompt_str + self.eot_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message, tools_prompt_str)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message, accumulate_reasoning=accumulate_reasoning)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message, tools_prompt_str=""):
        content = message["content"]

        if "# Tools" not in content and tools_prompt_str:
            content += tools_prompt_str

        return self.system_token + content + self.eot_token

    def parse_user(self, message):
        if "images" in message and message["images"] is not None:
            assert isinstance(message["images"], list), "images must be a list"
            n_imgs = len(message["images"])
            content = message["content"]
            if message["content"].startswith("<image>"):
                content = content[len("<image>") :]
            vision_tokens = (self.vision_start_token + self.image_token + self.vision_end_token) * n_imgs
            return self.user_token + vision_tokens + content + self.eot_token

        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message, accumulate_reasoning=False):
        content = (message.get("content", None) or "").strip()
        reasoning = (message.get("reasoning", None) or "").strip()
        tool_calls = message.get("tool_calls", None) or []

        if not reasoning and not tool_calls:
            return self.assistant_token + content + self.eot_token

        else:
            result = self.assistant_token

            if reasoning and accumulate_reasoning:
                result += "<think>\n" + reasoning + "\n</think>\n\n"

            if content:
                result += content
                if tool_calls:
                    result += "\n"

            if tool_calls:
                try:
                    tool_calls_strs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, ToolCall):
                            tool_call_dict = tool_call.to_dict()
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            tool_call_dict = tool_call["function"]
                        else:
                            tool_call_dict = tool_call
                        arguments_obj = tool_call_dict.get("arguments")
                        if isinstance(arguments_obj, str):
                            try:
                                arguments_obj = json.loads(arguments_obj)
                            except json.JSONDecodeError:
                                pass
                        tool_call_for_dump = dict(tool_call_dict)
                        if arguments_obj is not None:
                            tool_call_for_dump["arguments"] = arguments_obj
                        tool_call_str = f"{self.tool_parser.tool_call_begin}\n{json.dumps(tool_call_for_dump)}\n{self.tool_parser.tool_call_end}"
                        tool_calls_strs.append(tool_call_str)
                    tool_calls_str = "\n".join(tool_calls_strs)
                except Exception as e:
                    logger.error(f"Failed to format tool calls: {e}")
                    tool_calls_str = ""

                result += tool_calls_str

            result += self.eot_token
            return result

    def parse_tool(self, message):
        tool_outputs: list[ToolOutput | dict] = message.get("tool_outputs", [])

        if not tool_outputs:
            return self.user_token + self.tool_parser.tool_output_begin + "\n" + message["content"] + "\n" + self.tool_parser.tool_output_end + self.eot_token

        else:
            try:
                tool_outputs_strs = []
                for tool_output in tool_outputs:
                    if not isinstance(tool_output, ToolOutput):
                        tool_output = ToolOutput(**tool_output)
                    tool_output_str = f"{self.tool_parser.tool_output_begin}\n{str(tool_output)}\n{self.tool_parser.tool_output_end}"
                    tool_outputs_strs.append(tool_output_str)
                tool_outputs_str = "\n".join(tool_outputs_strs)
            except Exception as e:
                logger.error(f"Failed to format tool outputs: {e}")
                tool_outputs_str = ""

            return self.user_token + tool_outputs_str + self.eot_token

    def parse_completion(self, completion_ids):
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)

        if completion_text.count("</think>") == 1:
            reasoning, _, content = completion_text.partition("</think>")
            if reasoning.startswith("<think>"):
                reasoning = reasoning[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            reasoning = reasoning.strip()
            content = content.strip()
        else:
            reasoning = None
            content = completion_text
            if content.startswith("<think>"):
                content = content[len("<think>") :]
            if content.endswith(self.eos_token):
                content = content[: -len(self.eos_token)]
            if content.endswith(self.eot_token):
                content = content[: -len(self.eot_token)]
            content = content.strip()

        tool_calls = self.tool_parser.parse(content)

        begin_pattern = re.escape(self.tool_parser.tool_call_begin)
        end_pattern = re.escape(self.tool_parser.tool_call_end)
        content = re.sub(f"{begin_pattern}.*?{end_pattern}", "", content, flags=re.DOTALL)
        content = content.strip()

        return {
            "content": content,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        }

    def process_image_data(self, messages):
        from qwen_vl_utils import fetch_image

        messages = deepcopy(messages)
        image_data = []
        for message in messages:
            if "images" in message and message["images"] is not None:
                assert isinstance(message["images"], list), "images must be a list"
                images = message["images"]
                if not images or images[0] is None:
                    continue
                for image in images:
                    image_dict = image if isinstance(image, dict) else {"image": image}
                    processed_image = fetch_image(image_dict, image_patch_size=self.processor.image_processor.patch_size)  # PIL.Image.Image
                    image_data.append(processed_image)
        return image_data

    # ------------------------------------------------------------------
    # Multimodal cumulative tokenization (RL training path).
    #
    # tokenize_and_mask_cumulative re-tokenizes a whole multi-turn episode
    # TEXT-ONLY: each image collapses to a single unexpanded <|image_pad|> and
    # no pixel_values are produced, so the actor's training forward never sees
    # pixels (verified offline). This variant expands each <|image_pad|>
    # placeholder into its real patch tokens via the processor and returns
    # pixel_values / image_grid_thw, so the training forward conditions on the
    # same visual features the rollout saw.
    #
    # Key property (verified): the processor expands each image locally, so
    # per-segment expansion + concat is byte-identical to expanding the whole
    # rendered conversation at once. That lets us keep the segment-wise masking
    # of tokenize_and_mask_cumulative unchanged while emitting real pixels.
    # ------------------------------------------------------------------
    def _fetch_segment_images(self, msgs):
        """PIL images for the 'images' fields across msgs, in order (scoped
        version of process_image_data)."""
        from qwen_vl_utils import fetch_image

        out = []
        for m in msgs:
            imgs = m.get("images")
            if not imgs or imgs[0] is None:
                continue
            for image in imgs:
                image_dict = image if isinstance(image, dict) else {"image": image}
                out.append(fetch_image(image_dict, image_patch_size=self.processor.image_processor.patch_size))
        return out

    def _encode_segment_mm(self, text, seg_msgs):
        """Encode one rendered segment. If it contains <|image_pad|>
        placeholders, expand them via the processor and return per-image
        pixel_values / image_grid_thw.

        Returns (ids: list[int], per_img_pv: list[Tensor], per_img_thw: list[Tensor]).
        """
        n_ph = text.count(self.image_token)
        if n_ph == 0 or self.processor is None:
            return self.tokenizer.encode(text, add_special_tokens=False), [], []

        images = self._fetch_segment_images(seg_msgs)
        assert len(images) == n_ph, f"placeholder/image mismatch: {n_ph} <|image_pad|> vs {len(images)} images"
        mi = self.processor(text=[text], images=images, return_tensors="pt")
        ids = mi["input_ids"][0].tolist()
        pv = mi["pixel_values"]          # (sum t*h*w, patch_dim)
        thw = mi["image_grid_thw"]       # (n_ph, 3)
        counts = thw.prod(dim=1).tolist()
        per_pv, off = [], 0
        for c in counts:
            per_pv.append(pv[off:off + c])
            off += c
        per_thw = [thw[i:i + 1] for i in range(thw.shape[0])]
        return ids, per_pv, per_thw

    def _image_spans(self, ids, vs_id, ve_id):
        """Inclusive (start, end) index pairs of each
        <|vision_start|>..<|vision_end|> block, in order."""
        spans, start = [], None
        for i, t in enumerate(ids):
            if t == vs_id:
                start = i
            elif t == ve_id and start is not None:
                spans.append((start, i))
                start = None
        return spans

    def _span_safe_prefix(self, ids, per_pv, per_thw, keep_len, vs_id, ve_id):
        """Keep ids[:L] with L<=keep_len not splitting a vision block; keep only
        the images fully contained. Returns (ids, per_pv, per_thw)."""
        spans = self._image_spans(ids, vs_id, ve_id)
        L = keep_len
        for (s, e) in spans:
            if s < L <= e:      # cut would land inside this block -> drop it (and everything after)
                L = s
                break
        kept_imgs = sum(1 for (s, e) in spans if e < L)
        return ids[:L], per_pv[:kept_imgs], per_thw[:kept_imgs]

    def _span_safe_suffix(self, ids, per_pv, per_thw, keep_len, vs_id, ve_id):
        """Keep the rightmost ids[S:] (len<=keep_len) not splitting a vision
        block; drop leading images that fall out. Returns (ids, per_pv, per_thw)."""
        spans = self._image_spans(ids, vs_id, ve_id)
        S = len(ids) - keep_len
        for (s, e) in spans:
            if s < S <= e:      # cut inside this block -> move start past it
                S = e + 1
        kept = [i for i, (s, e) in enumerate(spans) if s >= S]
        n_drop = len(spans) - len(kept)
        return ids[S:], per_pv[n_drop:], per_thw[n_drop:]

    def tokenize_and_mask_cumulative_mm(self, messages, max_prompt_len=None, max_response_len=None):
        """Multimodal cumulative tokenization + image-span-aware truncation.

        Mirrors tokenize_and_mask_cumulative (same prompt/response split and
        assistant-only masking) but expands image placeholders into real patch
        tokens and returns multi_modal_inputs. When max_response_len is set and
        the response overflows, whole trailing turns (and their pixel_values)
        are dropped; a lone oversized first turn is cut at a vision-block
        boundary. Never splits a <|vision_start|>..<|vision_end|> block.

        Returns (prompt_ids, response_ids, response_mask, multi_modal_inputs).
        multi_modal_inputs is {} when there are no images (or no processor).
        """
        if self.processor is None:
            return (*self.tokenize_and_mask_cumulative(messages), {})

        vs_id = self.tokenizer.convert_tokens_to_ids(self.vision_start_token)
        ve_id = self.tokenizer.convert_tokens_to_ids(self.vision_end_token)
        merge_sq = int(getattr(self.processor.image_processor, "merge_size", 2)) ** 2

        try:
            first_assistant_idx = next(i for i, msg in enumerate(messages) if msg["role"] == "assistant")
        except StopIteration:
            raise ValueError("No assistant message found in chat_completions") from None

        # ---- PROMPT part (messages[:first_assistant_idx]) ----
        prompt_text = self.parse(messages[:first_assistant_idx], is_first_msg=True, add_generation_prompt=True, accumulate_reasoning=False)
        prompt_ids, prompt_pv, prompt_thw = self._encode_segment_mm(prompt_text, messages[:first_assistant_idx])

        # ---- RESPONSE segments (one per subsequent message) ----
        seg_ids, seg_mask, seg_pv, seg_thw = [], [], [], []
        for i in range(first_assistant_idx, len(messages)):
            if messages[i]["role"] == "assistant":
                text = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=False, accumulate_reasoning=True)
                text = text[len(self.generation_prompt):]
                ids, pv, thw = self._encode_segment_mm(text, [messages[i]])
                seg_ids.append(ids); seg_mask.append(1)
            else:
                text = self.parse([messages[i]], is_first_msg=False, add_generation_prompt=True, accumulate_reasoning=False)
                ids, pv, thw = self._encode_segment_mm(text, [messages[i]])
                seg_ids.append(ids); seg_mask.append(0)
            seg_pv.append(pv); seg_thw.append(thw)

        # ---- fit RESPONSE to max_response_len: drop WHOLE trailing turns ----
        if max_response_len is not None and sum(len(s) for s in seg_ids) > max_response_len:
            kept, acc = 0, 0
            for s in seg_ids:
                if acc + len(s) > max_response_len:
                    break
                acc += len(s); kept += 1
            if kept == 0:
                # first turn alone overflows: span-safe intra-turn cut
                kept = 1
                seg_ids[0], seg_pv[0], seg_thw[0] = self._span_safe_prefix(
                    seg_ids[0], seg_pv[0], seg_thw[0], max_response_len, vs_id, ve_id)
            if kept < len(seg_ids):
                logger.warning(f"cumulative_mm: response {sum(len(s) for s in seg_ids)} > {max_response_len}, dropped {len(seg_ids) - kept} trailing turn(s)")
            seg_ids, seg_mask = seg_ids[:kept], seg_mask[:kept]
            seg_pv, seg_thw = seg_pv[:kept], seg_thw[:kept]

        # ---- fit PROMPT (rare: rollout already enforces max_prompt at turn 0) ----
        if max_prompt_len is not None and len(prompt_ids) > max_prompt_len:
            logger.warning(f"cumulative_mm: prompt {len(prompt_ids)} > {max_prompt_len}, span-safe left truncation")
            prompt_ids, prompt_pv, prompt_thw = self._span_safe_suffix(
                prompt_ids, prompt_pv, prompt_thw, max_prompt_len, vs_id, ve_id)

        # ---- assemble ----
        response_ids, response_mask, resp_pv, resp_thw = [], [], [], []
        for ids, mv, pv, thw in zip(seg_ids, seg_mask, seg_pv, seg_thw):
            response_ids.extend(ids); response_mask.extend([mv] * len(ids))
            resp_pv.extend(pv); resp_thw.extend(thw)

        all_pv = list(prompt_pv) + resp_pv
        all_thw = list(prompt_thw) + resp_thw
        mm = {}
        if all_pv:
            mm["pixel_values"] = torch.cat(all_pv, dim=0)
            mm["image_grid_thw"] = torch.cat(all_thw, dim=0)

        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long)
        response_ids = torch.tensor(response_ids, dtype=torch.long)
        response_mask = torch.tensor(response_mask, dtype=torch.long)

        # ---- consistency: #image_pad must equal #image features ----
        if mm:
            pad_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            n_pad = int((prompt_ids == pad_id).sum() + (response_ids == pad_id).sum())
            n_feat = int((mm["image_grid_thw"].prod(dim=1) // merge_sq).sum())
            assert n_pad == n_feat, f"cumulative_mm pad/feature mismatch: {n_pad} != {n_feat}"

        return prompt_ids, response_ids, response_mask, mm


class LlamaChatTemplateParser(ChatTemplateParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.bos_token = "<|begin_of_text|>"
        self.system_token = "<|start_header_id|>system<|end_header_id|>\n\n"
        self.user_token = "<|start_header_id|>user<|end_header_id|>\n\n"
        self.assistant_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        self.eot_token = "<|eot_id|>"
        self.generation_prompt = self.assistant_token

        # tool tokens
        self.tool_start_token = "<|start_header_id|>tool<|end_header_id|>\n\n"
        self.tool_end_token = "<|eot_id|>"
        self.tool_response_start_token = "<|start_header_id|>tool_response<|end_header_id|>\n\n"
        self.tool_response_end_token = "<|eot_id|>"

        # TODO: add tool parser for llama

    def parse(self, messages, add_generation_prompt=False, is_first_msg=False, **kwargs) -> str:
        result = ""

        if is_first_msg:
            result += self.bos_token

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        return self.assistant_token + message["content"] + self.eot_token

    def parse_tool(self, message):
        return self.user_token + self.tool_response_start_token + message["content"] + self.tool_response_end_token + self.eot_token

    def parse_completion(self, completion_ids):
        # TODO: add parse_completion for llama
        raise NotImplementedError("LLamaChatTemplateParser does not support parse_completion")
