# SPDX-License-Identifier: Apache-2.0

"""Gemma3 token-type compatibility for the AReaL vision workflow."""

from __future__ import annotations

MODULE = "areal.workflow.vision_rlvr"


async def arun_episode(self, engine, data):
    """Run the v2.1 vision workflow with Gemma3 processor outputs."""
    from areal.workflow import vision_rlvr as upstream

    if isinstance(self.reward_fn, str):
        self.reward_fn = upstream.import_from_string(self.reward_fn)
        self.async_reward_fn = upstream.AsyncRewardWrapper(self.reward_fn)

    processed_input = self.processor(
        images=data["images"],
        text=data["messages"],
        padding=False,
        return_tensors="pt",
    )
    input_ids = processed_input["input_ids"].tolist()[0]
    if "mm_token_type_ids" not in processed_input:
        from transformers.models.gemma3.processing_gemma3 import Gemma3Processor

        if (
            isinstance(self.processor, Gemma3Processor)
            and "token_type_ids" in processed_input
        ):
            processed_input["mm_token_type_ids"] = processed_input["token_type_ids"]
    mm_token_type_ids = processed_input["mm_token_type_ids"].tolist()[0]

    req = upstream.ModelRequest(
        rid=upstream.uuid.uuid4().hex,
        input_ids=input_ids,
        image_data=upstream.image2base64(data["images"]),
        vision_msg_vllm=[data["messages_chat"]] if "messages_chat" in data else None,
        gconfig=self.gconfig.new(n_samples=1),
        tokenizer=self.tokenizer,
        processor=self.processor,
    )
    prompt_str = self.tokenizer.decode(input_ids)
    resp, reward = await self._collect_samples(engine, req, prompt_str, data)

    seq = resp.input_tokens + resp.output_tokens
    mm_token_type_ids += [0] * resp.output_len
    logprobs = [0.0] * resp.input_len + resp.output_logprobs
    loss_mask = [0] * resp.input_len + [1] * resp.output_len
    versions = [-1] * resp.input_len + resp.output_versions
    turn_ids = [-1] * resp.input_len + [0] * resp.output_len
    multi_modal_input = [{"pixel_values": processed_input["pixel_values"]}]
    if "image_grid_thw" in processed_input:
        multi_modal_input[0]["image_grid_thw"] = processed_input["image_grid_thw"]

    torch = upstream.torch
    return {
        "input_ids": torch.tensor(seq, dtype=torch.long).unsqueeze(0),
        "mm_token_type_ids": torch.tensor(
            mm_token_type_ids, dtype=torch.long
        ).unsqueeze(0),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
        "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
        "multi_modal_input": multi_modal_input,
        "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
        "turn_ids": torch.tensor(turn_ids, dtype=torch.int32).unsqueeze(0),
        "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
    }
