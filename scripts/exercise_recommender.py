def recommend_exercise(recovery_score):
    if recovery_score < 40:
        return "Try passive stretching exercises."
    elif recovery_score < 70:
        return "Do light arm raises and leg lifts."
    else:
        return "Perform resistance band exercises."