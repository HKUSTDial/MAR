import pandas as pd
import ast
import math
#
# Load the CSV file
df = pd.read_csv('../NewsPersonQA/output/qa_answer/final/all_results.csv', encoding='ISO-8859-1')

# Filter rows based on conditions
filtered_rows = df[(df['question_type'] == 'group') & (df['question_id'] % 2 == 1)]

# Iterate through filtered rows and compare 'standard_answer' with 'gpt4_answer'
for index, row in filtered_rows.iterrows():
    standard_answer = row['standard_answer']
    results = ast.literal_eval(standard_answer)

    df.at[index, 'results_num'] = len(results)

    gpt4_answer = row['gpt4_answer']
    llava7b_answer = row['7b+_answer'].replace('\\', '')
    llava13b_answer = row['13b+_answer'].replace('\\', '')
    mar_answer = row['mar_answers']

    cnt_gpt4 = 0
    cnt_7b = 0
    cnt_13b = 0
    cnt_mar = 0

    for result in results:
        try:
            if result in gpt4_answer:
                cnt_gpt4 += 1
        except:
            pass
        if result in llava7b_answer:
            cnt_7b += 1
        if result in llava13b_answer:
            cnt_13b += 1
        if result in mar_answer:
            cnt_mar += 1

    df.at[index, 'gpt4_correct_num'] = cnt_gpt4
    df.at[index, '7b+correct_num'] = cnt_7b
    df.at[index, '13b+correct_num'] = cnt_13b
    df.at[index, 'mar_correct_num'] = cnt_mar


# Save the DataFrame to a new CSV file
df.to_csv('../NewsPersonQA/output/qa_answer/final/all_results2.csv', index=False)
