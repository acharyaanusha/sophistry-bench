DEBATER_TURN_CLEAN = """
My answer is correct because the passage explicitly states this.
<claim>The expedition reached the summit on Tuesday.</claim>
<cite>The expedition reached the summit on Tuesday morning.</cite>
"""

DEBATER_TURN_NO_CITES = """
I assert this is correct.
<claim>This is true.</claim>
"""

DEBATER_TURN_MULTIPLE = """
<claim>Claim one.</claim>
<cite>Quote one.</cite>
<claim>Claim two.</claim>
<cite>Quote two.</cite>
<cite>Quote three.</cite>
"""

DEBATER_TURN_MALFORMED = """
<claim>Unclosed claim
<cite>Mismatched
"""
