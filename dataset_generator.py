"""
Dataset Generator — Creates synthetic labeled exam anxiety dataset
combining text responses + physiological features
"""
import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

LOW_PHRASES = [
    "I feel quite relaxed about my upcoming exams",
    "I have been studying consistently and feel confident",
    "I sleep well and feel mentally prepared",
    "Exams do not bother me much this time",
    "I am calm and focused during my study sessions",
    "I manage my time well and feel in control",
    "I feel a little nervous but it motivates me positively",
    "My preparation is going smoothly without much stress",
    "I enjoy studying and feel ready for assessments",
    "No major worries just some normal exam jitters",
    "I feel good about my chances in the exam",
    "I am confident in my preparation and feel steady",
]

MODERATE_PHRASES = [
    "I am somewhat nervous and find it hard to concentrate sometimes",
    "I worry a bit about failing but I keep studying",
    "I feel stressed but try to manage it with breaks",
    "Sometimes I cannot sleep because I keep thinking about exams",
    "I feel overwhelmed when I look at the entire syllabus",
    "I get anxious during revision but calm down after some time",
    "I have been feeling restless lately because of exams",
    "I feel moderate pressure from family expectations about results",
    "I get nervous but can still function and study normally",
    "My appetite has slightly reduced during exam preparation",
    "I feel tense when thinking about results but manage to study",
    "I sometimes procrastinate due to exam pressure and worry",
]

HIGH_PHRASES = [
    "I cannot stop worrying about failing the exam at all",
    "I feel extremely stressed and cannot focus on anything",
    "I experience panic attacks when I think about exams",
    "I have not been able to sleep for days because of exam fear",
    "I feel completely hopeless and unprepared for the exams",
    "My heart races and I feel nauseous when exam date approaches",
    "I cry frequently because of the unbearable exam pressure",
    "I feel like everything depends on this exam and I might fail",
    "I am constantly trembling and cannot eat or sleep properly",
    "My mind goes completely blank whenever I try to study",
    "The thought of the exam fills me with dread and terror",
    "I have been having severe headaches and chest tightness from stress",
]

def generate_physio(label):
    if label == 0:
        return {
            "heart_rate": round(np.random.normal(72, 5), 1),
            "sleep_hours": round(np.random.normal(7.5, 0.5), 1),
            "study_hours_per_day": round(np.random.normal(5, 1), 1),
            "days_since_last_break": random.randint(0, 2),
            "social_interactions_per_week": random.randint(5, 10),
            "water_intake_liters": round(np.random.normal(2.5, 0.3), 1),
            "exercise_minutes_per_day": random.randint(20, 60),
        }
    elif label == 1:
        return {
            "heart_rate": round(np.random.normal(83, 6), 1),
            "sleep_hours": round(np.random.normal(6.0, 0.7), 1),
            "study_hours_per_day": round(np.random.normal(7, 1), 1),
            "days_since_last_break": random.randint(2, 5),
            "social_interactions_per_week": random.randint(2, 6),
            "water_intake_liters": round(np.random.normal(1.8, 0.3), 1),
            "exercise_minutes_per_day": random.randint(5, 25),
        }
    else:
        return {
            "heart_rate": round(np.random.normal(96, 8), 1),
            "sleep_hours": round(np.random.normal(4.5, 0.8), 1),
            "study_hours_per_day": round(np.random.normal(9, 1.5), 1),
            "days_since_last_break": random.randint(5, 14),
            "social_interactions_per_week": random.randint(0, 2),
            "water_intake_liters": round(np.random.normal(1.2, 0.3), 1),
            "exercise_minutes_per_day": random.randint(0, 10),
        }

def build_dataset(n_per_class=400):
    rows = []
    pools = [LOW_PHRASES, MODERATE_PHRASES, HIGH_PHRASES]
    names = ["Low", "Moderate", "High"]
    for label_id, (phrases, name) in enumerate(zip(pools, names)):
        for _ in range(n_per_class):
            row = {
                "text_response": random.choice(phrases),
                "anxiety_label": name,
                "anxiety_code": label_id,
                **generate_physio(label_id),
            }
            rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

if __name__ == "__main__":
    df = build_dataset(400)
    df.to_csv("exam_anxiety.csv", index=False)
    print(f"✅ Dataset created → exam_anxiety.csv | Shape: {df.shape}")
    print(df["anxiety_label"].value_counts())