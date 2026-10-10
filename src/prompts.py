TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

def format_prompt(instruction: str) -> str:
    return TEMPLATE.format(instruction=instruction.strip())

