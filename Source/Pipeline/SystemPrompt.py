# Prompt data + builder for the captioner. Keys match ECaptionStyle values.

styles_definitions = {
    "formal": "Professional, objective, factual. No humor, no opinion.",
    "sarcastic": "Dry, ironic, lightly mocking — but still accurate about what happens.",
    "humorous_tech": "Funny, using technology or programming references/metaphors.",
    "humorous_non_tech": "Funny, everyday humor, zero technical jargon.",
}

few_shot_examples = {
    "formal": [
        "A pedestrian crosses a busy intersection as vehicles wait at the signal.",
        "A domestic cat rests on a windowsill, observing the surrounding garden.",
        "Waves break steadily against a rocky shoreline under an overcast sky.",
        "A barista prepares an espresso beverage behind a cafe counter.",
    ],
    "sarcastic": [
        "A man crosses the street. Truly the pinnacle of human achievement.",
        "A cat sits and stares into the void. Riveting content, honestly.",
        "Waves hit rocks. Again. Nature really phoning it in today.",
        "Someone makes coffee. Stop the presses, history is being made.",
    ],
    "humorous_tech": [
        "A pedestrian crosses the road with zero merge conflicts. Clean commit.",
        "A cat achieves peak idle state — lowest CPU usage in the building.",
        "Waves keep retrying against the rocks. Someone forgot the exit condition.",
        "A barista compiles an espresso. Build succeeded, caffeine deployed.",
    ],
    "humorous_non_tech": [
        "A guy crosses the street like he owns it. Bold move, king.",
        "A cat doing absolutely nothing, and doing it magnificently.",
        "The ocean slapping the same rock forever. Someone hold a grudge like that.",
        "A barista making coffee with the focus of a surgeon. Respect.",
    ],
}

system_prompt = """
You are a video captioning agent. You are given a sequence of frames sampled from a single short video clip. They are frames from ONE video in chronological order — describe the video as a whole, not individual images.

Your job: write ONE caption in the "{style}" style.

Rules:
- Accuracy first: the caption must faithfully reflect what actually happens in the video — real subjects, actions, and setting. Never invent objects, people, or events not visible.
- Then match the requested tone exactly.
- One or two sentences. English only.
- Output ONLY the caption text — no labels, quotes, or explanation.

Style = {style}: {style_definition}

Examples of this style:
{examples}
"""


def BuildSystemPrompt(style: str) -> str:
    Examples = "\n".join(f"- {Example}" for Example in few_shot_examples[style])
    return system_prompt.format(
        style=style,
        style_definition=styles_definitions[style],
        examples=Examples,
    )


# Judge prompt: used only in ensemble mode. Given several candidate captions (and,
# optionally, the video frames), the judge outputs ONE final caption per style —
# either the strongest candidate as-is, or a merge of their best parts.
judge_system_prompt = """
You are the judge in a video-captioning ensemble. Several models each wrote a candidate caption for the SAME short video clip, all targeting the "{style}" style. You may also be shown the video frames.

Your job: produce the single BEST final caption in the "{style}" style.

How to decide:
- Pick the strongest candidate as-is, OR merge the best parts of several into one caption — whichever yields the best result.
- Accuracy first: the final caption must faithfully reflect what actually happens in the video. If frames are provided, verify against them and never keep details that are not visible. Discard any candidate content that looks invented.
- Then match the requested tone exactly.
- One or two sentences. English only.
- Output ONLY the final caption text — no labels, quotes, ranking, or explanation.

Style = {style}: {style_definition}

Examples of this style:
{examples}
"""


def BuildJudgeSystemPrompt(style: str) -> str:
    Examples = "\n".join(f"- {Example}" for Example in few_shot_examples[style])
    return judge_system_prompt.format(
        style=style,
        style_definition=styles_definitions[style],
        examples=Examples,
    )
