"""
XGBoost Model for Datathon 2026
Goal: Beat 13.20% engagement rate on test set

Author: Alexander
"""

#imports
import polars as pl
import numpy as np
import xgboost as xgb
import sklearn
import random
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

def get_days_since_shown(row):
    """
    Extract days since the selected template was last shown.
    
    Args:
        row: Dict with 'selected_template' and 'history' keys
    
    Returns:
        float: Minimum n_days for matching template, or 999.0 if never shown
    """
    template = row['selected_template']
    history = row['history']
    
    # Find all history entries where template matches
    matching_days = [float(h['n_days']) for h in history if h['template'] == template]
    
    # Return minimum (most recent) or 999.0 if never shown
    return float(min(matching_days)) if matching_days else 999.0

# TODO: Load data
# - Load train_sample.parquet (100K rows for fast iteration)
# - Load val.parquet for validation
# - Check shapes and engagement rates
pl.set_random_seed(42)
# Load pre-sampled data (random samples created by process_data.py)
df = pl.read_parquet("Data/train_rsample_100k.parquet")
df_val = pl.read_parquet("Data/val_rsample_100k.parquet")

print(f"Train shape: {df.shape}")
print(f"Val shape: {df_val.shape}")
print(f"Train columns {df.collect_schema}")
# TODO: Add days_since_shown feature
# - Import or define get_days_since_shown function
# - Apply to both train and val datasets
# - Verify it worked (check column exists and has reasonable values)
print("Adding days_since_shown feature...")
df = df.with_columns(pl.struct(['selected_template', 'history']).map_elements(get_days_since_shown, return_dtype=pl.Float64).alias('days_since_shown'))
df_val = df_val.with_columns(pl.struct(['selected_template', 'history']).map_elements(get_days_since_shown, return_dtype=pl.Float64).alias('days_since_shown'))
print("days_since_shown added. Sample values:")
print(df.select('days_since_shown').head(5))


df_pd = df.to_pandas()
df_val_pd = df_val.to_pandas()

#Changing time of day to it's actual intended feature, was just a copy of datetime for now
df_pd['time_of_day'] = df_pd['datetime'] - np.floor(df_pd['datetime'])
df_val_pd['time_of_day'] = df_val_pd['datetime'] - np.floor(df_val_pd['datetime'])
print(f"time_of_day sample after fix: {df_pd['time_of_day'].head()}")

#encode categorical features
print("Encoding categorical features...")
le_template = LabelEncoder()
le_language = LabelEncoder()

df_pd['template_encoded'] = le_template.fit_transform(df_pd['selected_template'])
df_pd['language_encoded'] = le_language.fit_transform(df_pd['ui_language'])

df_val_pd['template_encoded'] = le_template.transform(df_val_pd['selected_template'])
df_val_pd['language_encoded'] = le_language.transform(df_val_pd['ui_language'])

print("Encoded selected_template and ui_language.")

feature_cols = ['template_encoded', 'language_encoded', 'days_since_shown', 'history_length', 'n_eligible', 'time_of_day']

X_train = df_pd[feature_cols]
y_train = df_pd['session_end_completed']

X_val = df_val_pd[feature_cols]
y_val = df_val_pd['session_end_completed']

print(f"\n Features used for modeling: {feature_cols}")


#Training XGBoost model

print("\nTraining XGBoost model...")

#we should find the positive class weight to balance the dataset

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight: {pos_weight:.2f}")
#for simplicity, we will use default hyperparameters for now. We can tune them later if needed.
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, scale_pos_weight=pos_weight,random_state=42)
#train the model on our training data
model.fit(X_train, y_train)

probs = model.predict_proba(X_val)[:, 1]  # Get probabilities for class 1 (engaged)



#OFF POLICY EVALUATION
#We can only validate when our choice actually matches what was sent
correct_choices = 0 #number of times our model's top predicted template matches the actual selected_template
successes = 0 #number of times our model's top predicted template matches the actual selected_template and the user engaged

for i in range(len(df_val_pd)):
    row = df_val_pd.iloc[i]
    eligible = row['eligible_templates'] #list of eligible templates for this notification
    actual_template = row['selected_template'] #the template that was actually sent
    actual_engagement = row['session_end_completed'] #whether the user engaged (1) or not (0)

    #predict prob for each eligible template
    best_prob = -1
    best_template = None

    for template in eligible:
        #Create a copy of row with this template
        temp_row = row.copy()
        temp_row['selected_template'] = template
        temp_row['template_encoded'] = le_template.transform([template])[0] #encode the template

        #Recompute days_since_shown for this template
        temp_row['days_since_shown'] = get_days_since_shown(temp_row)

        #predict
        X_test = temp_row[feature_cols].values.reshape(1, -1)
        prob = model.predict_proba(X_test)[0,1] #probability of engagement

        if prob > best_prob:
            best_prob = prob
            best_template = template

    #Check if our best template matches the actual template and if the user engaged
    if best_template == actual_template:
        correct_choices += 1
        if actual_engagement == 1:
            successes += 1

print(f"\nOff-policy evaluation results:")
policy_engagement_rate = successes / correct_choices if correct_choices > 0 else 0
print(f"Our model engagement rate (off-policy): {policy_engagement_rate:.4%}")
print(f"Baseline rate: {y_train.mean():.4%} (engagement rate in training set)")
print(f"Validation set engagement rate: {y_val.mean():.4%}")
print(f"Number of matching decisions: {correct_choices} / {len(df_val_pd)} ({correct_choices/len(df_val_pd):.2%})")

#These results are quite suspicious, the base rate for the validation set is already 18.4%, suggesting that there might be something wrong with our train/val split
#On top of that ,the engagement rate we find is extremely high, likely due to the case that we more often have matches with low template_options, which might imply that the user is already more engaged.
#We should check if this bias exists by checking the correlation between number of eligible templates and engagement rate.

# First, we check if theres a temporal change in the engagement rate in general, to check if something is wrong with our train/val split

#Group by day and calculate engagement rate
engagement_by_day = df_pd.groupby(df_pd['datetime'].astype(int))['session_end_completed'].agg(['mean', 'count'])

print("\nEngagement rate by day")
print(engagement_by_day)

#Plot engagement rate by day
plt.figure(figsize=(10,5))
plt.plot(engagement_by_day.index, engagement_by_day['mean'], marker='o')
plt.xlabel('Day (since experiment start)')
plt.ylabel('Engagement Rate')
plt.title('Engagement Rate Over Time in Training Set')
plt.axhline(y=df_pd['session_end_completed'].mean(), color='r', linestyle='--', label=f'Overall Mean ({df_pd["session_end_completed"].mean():.2%})')
plt.legend()
plt.grid(True)
plt.show()


# TODO: Feature importance
# - Extract and print feature importances
# - Understand what drives engagement

# TODO: Hyperparameter tuning (if time)
# - Try different learning rates
# - Try different tree depths
# - Use GridSearchCV or RandomizedSearchCV

# TODO: Save model
# - Pickle or save model for later use
# - Document final hyperparameters and performance

# TODO: Test on actual test set (ONLY ONCE at the very end)
# - Load test data
# - Apply same preprocessing
# - Evaluate final performance
# - This is the number we report!
