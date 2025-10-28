import os
import re
import pandas as pd
import numpy as np


def extract_summary_data(summary_file_path):
    """Extract key metrics from a summary.txt file."""
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    data = {}

    # Extract function name
    func_match = re.search(r'Functie:\s*(\w+)', content)
    data['Function'] = func_match.group(1) if func_match else 'Unknown'

    # Extract dimension
    dim_match = re.search(r'Dimensiune:\s*(\d+)', content)
    data['Dimension'] = int(dim_match.group(1)) if dim_match else None

    # Extract selection method
    sel_match = re.search(r'Selectie:\s*(\w+)', content)
    data['Selection'] = sel_match.group(1) if sel_match else 'Unknown'

    # Extract crossover rate
    cross_match = re.search(r'r_cross:\s*([\d.]+)', content)
    data['Crossover_Rate'] = float(cross_match.group(1)) if cross_match else None

    # Extract mutation multiplier
    mut_match = re.search(r'Multiplier:\s*([\d.]+)', content)
    data['Mutation_Multiplier'] = float(mut_match.group(1)) if mut_match else None

    # Extract initial mutation rate
    mut_init_match = re.search(r'r_mut_initial:\s*([\d.]+)', content)
    data['Initial_Mutation_Rate'] = float(mut_init_match.group(1)) if mut_init_match else None

    # Extract iterations
    iter_match = re.search(r'n_iter:\s*(\d+)', content)
    data['Iterations'] = int(iter_match.group(1)) if iter_match else None

    # Extract population size
    pop_match = re.search(r'n_pop:\s*(\d+)', content)
    data['Population_Size'] = int(pop_match.group(1)) if pop_match else None

    # Extract stagnation limit
    stag_match = re.search(r'stagnation_limit:\s*(\d+|None)', content)
    if stag_match:
        stag_val = stag_match.group(1)
        data['Stagnation_Limit'] = int(stag_val) if stag_val != 'None' else None
    else:
        data['Stagnation_Limit'] = None

    # Extract number of runs
    runs_match = re.search(r'Numar de rulari:\s*(\d+)', content)
    data['Number_of_Runs'] = int(runs_match.group(1)) if runs_match else None

    # Extract average execution time
    time_match = re.search(r'Timp mediu de executie:\s*([\d.]+)', content)
    data['Avg_Time_Seconds'] = float(time_match.group(1)) if time_match else None

    # Extract best score
    best_match = re.search(r'Cel mai bun scor:\s*([\d.eE+-]+)', content)
    data['Best_Score'] = float(best_match.group(1)) if best_match else None

    # Extract mean score
    mean_match = re.search(r'Scorul mediu:\s*([\d.eE+-]+)', content)
    data['Mean_Score'] = float(mean_match.group(1)) if mean_match else None

    # Extract standard deviation
    std_match = re.search(r'Dev. standard:\s*([\d.eE+-]+)', content)
    data['Std_Dev'] = float(std_match.group(1)) if std_match else None

    return data


def aggregate_all_results(main_output_dir):

    all_results = []

    if not os.path.exists(main_output_dir):
        print(f"Directory {main_output_dir} does not exist!")
        return pd.DataFrame()

    # Iterate through all subdirectories
    for exp_dir in os.listdir(main_output_dir):
        exp_path = os.path.join(main_output_dir, exp_dir)

        if os.path.isdir(exp_path):
            summary_file = os.path.join(exp_path, 'summary.txt')

            if os.path.exists(summary_file):
                try:
                    data = extract_summary_data(summary_file)
                    data['Experiment_Name'] = exp_dir
                    all_results.append(data)
                    print(f"Processed: {exp_dir}")
                except Exception as e:
                    print(f"Error processing {exp_dir}: {e}")

    if not all_results:
        print("No results found!")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Reorder columns for better readability
    column_order = [
        'Experiment_Name', 'Function', 'Dimension', 'Selection',
        'Crossover_Rate', 'Mutation_Multiplier', 'Initial_Mutation_Rate',
        'Best_Score', 'Mean_Score', 'Std_Dev',
        'Iterations', 'Population_Size', 'Stagnation_Limit',
        'Number_of_Runs', 'Avg_Time_Seconds'
    ]

    df = df[[col for col in column_order if col in df.columns]]

    return df


def create_summary_statistics(df):

    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(['Function', 'Dimension']).agg({
        'Best_Score': ['min', 'mean', 'std'],
        'Mean_Score': ['min', 'mean', 'std'],
        'Avg_Time_Seconds': 'mean'
    }).round(6)

    return summary


def find_best_configurations(df, top_n=5):

    if df.empty:
        return pd.DataFrame()

    best_configs = []

    for (func, dim), group in df.groupby(['Function', 'Dimension']):
        # Sort by best score
        top_configs = group.nsmallest(top_n, 'Best_Score')
        best_configs.append(top_configs)

    return pd.concat(best_configs, ignore_index=True)


if __name__ == '__main__':
    # Directory containing your results
    MAIN_OUTPUT_DIR = "GRID_SEARCH_RESULTS_BINAR_IMBUNATATIT"

    print("Starting aggregation of results...\n")

    # Aggregate all results
    df_results = aggregate_all_results(MAIN_OUTPUT_DIR)

    if not df_results.empty:
        # Save complete results
        output_file = "aggregated_summary_binary_improved.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\n✓ Complete results saved to: {output_file}")
        print(f"  Total experiments: {len(df_results)}")

        # Create and save summary statistics
        df_summary = create_summary_statistics(df_results)
        summary_file = "summary_statistics.csv"
        df_summary.to_csv(summary_file)
        print(f"\n✓ Summary statistics saved to: {summary_file}")

        # Find and save best configurations
        df_best = find_best_configurations(df_results, top_n=5)
        best_file = "best_configurations.csv"
        df_best.to_csv(best_file, index=False)
        print(f"\n✓ Best configurations saved to: {best_file}")

        # Display some statistics
        print("\n" + "=" * 60)
        print("QUICK SUMMARY")
        print("=" * 60)
        print(f"\nFunctions tested: {df_results['Function'].unique()}")
        print(f"Dimensions tested: {sorted(df_results['Dimension'].unique())}")
        print(f"Selection methods: {df_results['Selection'].unique()}")

        print("\n" + "=" * 60)
        print("BEST SCORE PER FUNCTION-DIMENSION")
        print("=" * 60)
        best_per_group = df_results.groupby(['Function', 'Dimension'])['Best_Score'].min()
        print(best_per_group)

    else:
        print("\n⚠ No results were found or processed.")
        print(f"  Make sure the directory '{MAIN_OUTPUT_DIR}' exists and contains experiment results.")