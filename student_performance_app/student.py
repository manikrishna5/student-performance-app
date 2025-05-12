import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
df = pd.read_csv(r"C:\Users\Mani Krishna Karri\student_performance_app\student_habits_performance.csv")

df['parental_education_level'] = df['parental_education_level'].fillna('Missing')
df['parental_education_level'] = (df['parental_education_level']=='Missing').astype(int)
df['extra_wastage_time'] = df['social_media_hours']+df['netflix_hours']
df = df.drop('social_media_hours',axis=1)
df = df.drop('netflix_hours',axis=1)
df['part_time_job'] = (df['part_time_job']=='Yes').astype(int)
df['extracurricular_participation'] = (df['extracurricular_participation']=='Yes').astype(int)
df = pd.get_dummies(df,columns=['gender'],drop_first=True)
df = df.drop('student_id',axis=1)
diet_mapping = {'Poor':0,'Fair':1,'Good':2}
df['diet_quality'] = df['diet_quality'].map(diet_mapping)
internet_mapping = {'Poor': 0, 'Average': 1, 'Good': 2}
df['internet_quality'] = df['internet_quality'].map(internet_mapping)
#scale
scaler = StandardScaler()

cols_to_scale = [
    'age', 'study_hours_per_day', 'attendance_percentage',
    'sleep_hours', 'extra_wastage_time', 'mental_health_rating'
]
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
# df.to_csv('student_habits_preprocessed.csv', index=False)
# df.to_csv('new_student_habits_preprocessed.csv', index=False)
#train_test
x = df.drop('exam_score',axis=1)
y = df['exam_score']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(mse)
print(r2)

joblib.dump(reg, 'linear_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


