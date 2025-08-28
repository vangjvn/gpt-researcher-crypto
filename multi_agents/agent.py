from multi_agents.agents import ChiefEditorAgent

chief_editor = ChiefEditorAgent({
  "query": "Is AI in a hype cycle?",
  "max_sections": 3,
  "follow_guidelines": False,
  "model": "gpt-4o-mini",
  "guidelines": [
    "The report MUST be written in APA format",
    "Each sub section MUST include supporting sources using hyperlinks. If none exist, erase the sub section or rewrite it to be a part of the previous section",
    "识别当前任务所用语言，报告必须跟随当前任务所用的语言"
  ],
  "verbose": False
}, websocket=None, stream_output=None)
graph = chief_editor.init_research_team()
graph = graph.compile()