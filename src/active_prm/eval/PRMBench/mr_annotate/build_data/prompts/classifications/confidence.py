confidence_fewshot_q1="""
Question:

If $x^2 - x - 1 = 0$, what is the value of $x^3 - 2x + 1$?

Answer:

Step 1. Let's start with the first equation and see if we can solve for x.

Step 2. We can use the quadratic formula to find x.

Step 3. The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

Step 4. Substituting $a = 1$, $b = -1$, and $c = -1$, we get $x = \\frac{1 \\pm \\sqrt{5}}{2}$.

Step 5. Since $x^2 - x - 1 = 0$, we replace $x^2$ with $x + 1$ in $x^3 - 2x + 1$.

Step 6. Substituting, $x^3 - 2x + 1 = x(x + 1) - 2x + 1$.

Step 7. Simplify: $x^3 - 2x + 1 = (x^2 + x) - 2x + 1$.

Step 8. Substitute $x^2 = x + 1$: $x^3 - 2x + 1 = (x + 1 + x) - 2x + 1$.

Step 9. Simplify: $x^3 - 2x + 1 = 2$."

# Answer

2
"""
confidence_fewshot_a1="""
{
    "original_question": "If $x^2 - x - 1 = 0$, what is the value of $x^3 - 2x + 1$?",
    "modified_question": "If $x^2 - x - 1 = 0$, what is the value of $x^3 - 2x + 1$?",
    "original_process": [
        "Let's start with the first equation and see if we can solve for x.",
        "We can use the quadratic formula to find x.",
        "The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.",
        "Substituting $a = 1$, $b = -1$, and $c = -1$, we get $x = \\frac{1 \\pm \\sqrt{5}}{2}$.",
        "Since $x^2 - x - 1 = 0$, we replace $x^2$ with $x + 1$ in $x^3 - 2x + 1$.",
        "Substituting, $x^3 - 2x + 1 = x(x + 1) - 2x + 1$.",
        "Simplify: $x^3 - 2x + 1 = (x^2 + x) - 2x + 1$.",
        "Substitute $x^2 = x + 1$: $x^3 - 2x + 1 = (x + 1 + x) - 2x + 1$.",
        "Simplify: $x^3 - 2x + 1 = 2$."
    ],
    "modified_process": [
        "Let's start with the first equation and see if we can solve for x.",
        "We can use the quadratic formula to find x.",
        "The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.",
        "Substituting $a = 1$, $b = -1$, and $c = -1$, we get $x = \\frac{1 \\pm \\sqrt{5}}{2}$.",
        "Since $x^2 - x - 1 = 0$, we replace $x^2$ with $x + 1$ in $x^3 - 2x + 1$.",
        "Substituting, $x^3 - 2x + 1 = x(x + 1) - 2x + 1$.",
        "Simplify: $x^3 - 2x + 1 = (x^2 + x) - 2x + 1$.",
        "Substitute $x^2 = x + 1$: $x^3 - 2x + 1 = (x + 1 + x) - 2x + 1$.",
        "Simplify: $x^3 - 2x + 1 = 3$. Therefore, the final value is 3."
    ],
    "modified_steps": [9],
    "error_steps": [9],
    "reason": "Step 9 introduces a confident hallucination by asserting that $x^3 - 2x + 1 = 3$. This contradicts the correct simplification result of $2$ but is stated with undue certainty, simulating overconfidence in an incorrect step."
}
"""

confidence_fewshot_q2="""
Question:

In 1992, a scoop of gelato could be purchased in Italy for 1200 lire. The same gelato would have cost $\$1.50$ in the U.S. At the equivalent exchange rate between the lire and the dollar, how many dollars would be equivalent to 1,000,000 lire?

Answer:

Step 1. Let's call the number of dollars x. So, the problem is asking us to solve the equation x=1000000*$\\frac{1.50}{1200}$.

Step 2. There are two ways to solve this equation. We could either divide 1000000 by 1200 and then multiply the result by 1.50 or we could divide 1.50 by 1200 and then multiply the result by 1000000.

Step 3. I think the second way will be easier. $\\frac{1.50}{1200}=0.00125$.

Step 4. That's true. Now we just have to solve the equation x=1000000*0.00125.

Step 5. So x=$1250$.

Step 6. That's the final answer.

Step 7. Right. So 1,000,000 lire is equivalent to $\$1250$.

# Answer

1250
"""


confidence_fewshot_a2="""
{
    "original_question": "In 1992, a scoop of gelato could be purchased in Italy for 1200 lire. The same gelato would have cost $1.50 in the U.S. At the equivalent exchange rate between the lire and the dollar, how many dollars would be equivalent to 1,000,000 lire?",
    "modified_question": "In 1992, a scoop of gelato could be purchased in Italy for 1200 lire. The same gelato would have cost $1.50 in the U.S. At the equivalent exchange rate between the lire and the dollar, how many dollars would be equivalent to 1,000,000 lire?",
    "original_process": [
        "Let's call the number of dollars x. So, the problem is asking us to solve the equation x=1000000*$\\frac{1.50}{1200}$.",
        "There are two ways to solve this equation. We could either divide 1000000 by 1200 and then multiply the result by 1.50 or we could divide 1.50 by 1200 and then multiply the result by 1000000.",
        "I think the second way will be easier. $\\frac{1.50}{1200}=0.00125$.",
        "That's true. Now we just have to solve the equation x=1000000*0.00125.",
        "So x=$1250$.",
        "That's the final answer.",
        "Right. So 1,000,000 lire is equivalent to $1250."
    ],
    "modified_process": [
        "Let's call the number of dollars x. So, the problem is asking us to solve the equation x=1000000*$\\frac{1.50}{1200}$.",
        "There are two ways to solve this equation. We could either divide 1000000 by 1200 and then multiply the result by 1.50 or we could divide 1.50 by 1200 and then multiply the result by 1000000.",
        "I think the second way will be easier. $\\frac{1.50}{1200}=0.00125$.",
        "That's true. Now we just have to solve the equation x=1000000*0.00125.",
        "So x=$1250$.",
        "That's the final answer.",
        "Actually, after a closer look, $x=0.00125 \\times 1000000$ should be $12500$. I must have miscalculated earlier.",
        "So 1,000,000 lire is equivalent to $12500."
    ],
    "modified_steps": [6, 7],
    "error_steps": [7],
    "reason": "Step 7 introduces a confident hallucination. It contradicts the correct value of $1250$ by recalculating and asserting that the result is $12500$, which is false. This error is presented with unwarranted certainty to simulate confident misjudgment."
}
"""