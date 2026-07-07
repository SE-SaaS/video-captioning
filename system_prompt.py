# The styles definitions 
formal = "Professional, objective, factual. No humor, no opinion."
sarcastic = "Dry, ironic, lightly mocking — but still accurate about what happens."
humorous_tech = "Funny, using technology or programming references/metaphors."
humorous_non_tech = "Funny, everyday humor, zero technical jargon."

styles_definitions = {
    "formal": "Professional, objective, factual. No humor, no opinion.",
    "sarcastic": "Dry, ironic, lightly mocking — but still accurate about what happens.",
    "humorous_tech": "Funny, using technology or programming references/metaphors.",
    "humorous_non_tech": "Funny, everyday humor, zero technical jargon."
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

Style = {style}: {styles_definitions[style]}

Examples of this style:
{few_shot_examples[style]}
"""