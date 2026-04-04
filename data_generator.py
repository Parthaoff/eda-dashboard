import pandas as pd
import numpy as np
np.random.seed(42)
n = 200

data = {
    "Student_ID": range(1, n+1),
    "Name": [f"Student_{i}" for i in range(1, n+1)],
    "Gender": np.random.choice(["Male", "Female"], n),
    "Subject": np.random.choice(["Math", "Science", "English", "History"], n),
    "Marks_Obtained": np.random.randint(30, 100, n),
    "Total_Marks": 100,
    "Attendance_Percentage": np.random.randint(50, 100, n),
}

df = pd.DataFrame(data)
def grade(marks):
    if marks >= 90: return "A"
    elif marks >= 75: return "B"
    elif marks >= 60: return "C"
    elif marks >= 40: return "D"
    else: return "F"

df["Grade"] = df["Marks_Obtained"].apply(grade)

df.to_csv("student_dataset.csv", index=False)
print("Dataset Created!")