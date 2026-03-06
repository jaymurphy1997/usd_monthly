import pandas as pd
from typing import List, Dict, Tuple

# Define the official program order based on the monthly report
PROGRAM_ORDER = [
    'Informatics',
    'CyberOps',
    'CyberEng',
    'MED',
    'LEPSL',
    'Data Science',
    'MSAAI',
    'MSLDT',
    'MTS',
    'MESH',
    'MSHA',
    'MSNP',
    'EML',
    'MSITL',
    'MSNNL'
]

# Program display names mapping
PROGRAM_NAMES = {
    'Informatics': 'Health Care Informatics',
    'CyberOps': 'Cyber Security Operations and Leadership',
    'CyberEng': 'Cyber Security Engineering',
    'MED': 'Education',
    'LEPSL': 'Law Enforcement and Public Safety Leadership',
    'Data Science': 'Applied Data Science',
    'MSAAI': 'Applied Artificial Intelligence',
    'MSLDT': 'Learning Design and Technology',
    'MTS': 'Theological Studies',
    'MESH': 'Engineering for Sustainability & Health',
    'MSHA': 'Humanitarian Action',
    'MSNP': 'Nonprofit Leadership & Management',
    'EML': 'Engineering Management and Leadership',
    'MSITL': 'Information Technology Leadership',
    'MSNNL': 'MSN Nursing Leadership'
}


def load_data(mom_file: str, yoy_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MoM and YoY data from CSV files."""
    mom_data = pd.read_csv(mom_file)
    yoy_data = pd.read_csv(yoy_file)
    return mom_data, yoy_data


def filter_channels(data: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    """Filter data by specified channels."""
    return data[data['default_channel'].isin(channels)].copy()


def get_top_pages(data: pd.DataFrame, group_cols: List[str],
                  metric: str, top_n: int = 3) -> pd.DataFrame:
    """Get top N pages by metric for each program and channel combination."""
    result = (data.groupby(group_cols)
              .apply(lambda x: x.nlargest(top_n, metric), include_groups=False)
              .reset_index())
    # Drop the extra level index column created by groupby apply
    if 'level_2' in result.columns:
        result = result.drop(columns=['level_2'])
    elif 'level_1' in result.columns:
        result = result.drop(columns=['level_1'])
    return result


def sort_by_program_order(data: pd.DataFrame) -> pd.DataFrame:
    """Sort data according to the official program order."""
    # Create a categorical type with the specified order
    data['program_category'] = pd.Categorical(
        data['program_category'],
        categories=PROGRAM_ORDER,
        ordered=True
    )
    # Sort by program_category and default_channel
    return data.sort_values(['program_category', 'default_channel']).reset_index()


def summarize_data(mom_data: pd.DataFrame, yoy_data: pd.DataFrame) -> pd.DataFrame:
    """Merge MoM and YoY data and prepare summary."""
    summary = pd.merge(
        mom_data,
        yoy_data,
        on=['program_category', 'default_channel', 'Landing_page'],
        how='outer',
        suffixes=('_mom', '_yoy')
    )
    
    # Fill NaN values
    summary = summary.fillna({
        'Session_mom': 0,
        'Session_yoy': 0,
        'sessions_mom_difference': 0,
        'sessions_yoy_difference': 0,
        'Conversions_mom': 0,
        'Conversions_yoy': 0,
        'conversions_mom_difference': 0,
        'conversions_yoy_difference': 0,
        'conversion_rate_mom_percent_difference': 0,
        'conversion_rate_yoy_percent_difference': 0
    })
    
    # Sort by the official program order
    summary = sort_by_program_order(summary)
    
    return summary


def generate_markdown(data: pd.DataFrame, output_file: str) -> None:
    """Generate markdown report with programs in official order."""
    
    # Filter out pages with 20 or fewer sessions
    data = data[data['Session_mom'] > 20].copy()
    
    # Ensure data is sorted by program order
    data = sort_by_program_order(data)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        
        # Group by program (maintaining order)
        for program in PROGRAM_ORDER:
            program_data = data[data['program_category'] == program]
            
            if program_data.empty:
                continue
                
            # Get display name
            display_name = PROGRAM_NAMES.get(program, program)
            f.write(f"## **{display_name}**\n\n")
            
            f.write(f"### **Landing Page Report**\n\n")
            
            # Group by channel
            for channel in program_data['default_channel'].unique():
                channel_data = program_data[program_data['default_channel'] == channel].sort_values('Session_mom', ascending=False)
                
                f.write(f"#### **{channel}**\n\n")
                
                for _, row in channel_data.iterrows():
                    page = row['Landing_page']
                    
                    # Sessions
                    sessions_mom = row.get('Session_mom', 0)
                    sessions_mom_diff = row.get('sessions_mom_difference', 0)
                    sessions_yoy_diff = row.get('sessions_yoy_difference', 0)
                    
                    # Conversions
                    conv_mom = row.get('Conversions_mom', 0)
                    conv_mom_diff = row.get('conversions_mom_difference', 0)
                    conv_yoy_diff = row.get('conversions_yoy_difference', 0)

                    # Calculate percent change in conversions (not conversion rate)
                    # Previous period conversions = current - difference
                    prev_conv_mom = conv_mom - conv_mom_diff
                    prev_conv_yoy = conv_mom - conv_yoy_diff

                    # Calculate percentage change (handle division by zero)
                    if prev_conv_mom != 0:
                        conv_pct_change_mom = (conv_mom_diff / prev_conv_mom) * 100
                    else:
                        conv_pct_change_mom = 100.0 if conv_mom_diff > 0 else 0.0

                    if prev_conv_yoy != 0:
                        conv_pct_change_yoy = (conv_yoy_diff / prev_conv_yoy) * 100
                    else:
                        conv_pct_change_yoy = 100.0 if conv_yoy_diff > 0 else 0.0
                    
                    # Format the output with proper signs
                    sessions_yoy_str = f"+{sessions_yoy_diff:.0f}" if sessions_yoy_diff >= 0 else f"{sessions_yoy_diff:.0f}"
                    sessions_mom_str = f"+{sessions_mom_diff:.0f}" if sessions_mom_diff >= 0 else f"{sessions_mom_diff:.0f}"
                    
                    conv_yoy_diff_str = f"+{conv_yoy_diff:.0f}" if conv_yoy_diff >= 0 else f"{conv_yoy_diff:.0f}"
                    conv_yoy_pct_str = f"+{conv_pct_change_yoy:.0f}" if conv_pct_change_yoy >= 0 else f"{conv_pct_change_yoy:.0f}"

                    conv_mom_diff_str = f"+{conv_mom_diff:.0f}" if conv_mom_diff >= 0 else f"{conv_mom_diff:.0f}"
                    conv_mom_pct_str = f"+{conv_pct_change_mom:.0f}" if conv_pct_change_mom >= 0 else f"{conv_pct_change_mom:.0f}"
                    
                    # Write the bullet point
                    f.write(f'* "{page}": {sessions_mom:.0f} sessions ')
                    f.write(f'(YoY: {sessions_yoy_str} | MoM: {sessions_mom_str})')
                    
                    # Only show conversions if there are any
                    if conv_mom > 0:
                        f.write(f' Conversions: {conv_mom:.0f} ')
                        f.write(f'(YoY: {conv_yoy_diff_str}, {conv_yoy_pct_str}% | ')
                        f.write(f'MoM: {conv_mom_diff_str}, {conv_mom_pct_str}%)')
                    
                    f.write('\n')
                
                f.write("\n")
            
            f.write("\n")
    
    print(f"Markdown file created successfully with platform-independent newlines: {output_file}")


if __name__ == "__main__":
    # File paths
    mom_file = 'USDOnline-DashboardMk2_RLandingPagesMoM_Table.csv'
    yoy_file = 'USDOnline-DashboardMk2_RLandingPagesYoY_Table.csv'
    channels_of_interest = ['Organic Search', 'Paid Search', 'Paid Social']
    
    # Load data
    mom_data, yoy_data = load_data(mom_file, yoy_file)
    
    # Diagnostic: Check what columns actually exist
    print("MoM columns:", mom_data.columns.tolist())
    print("YoY columns:", yoy_data.columns.tolist())
    print("MoM shape:", mom_data.shape)
    print("YoY shape:", yoy_data.shape)
    
    # Filter data
    mom_filtered = filter_channels(mom_data, channels_of_interest)
    yoy_filtered = filter_channels(yoy_data, channels_of_interest)
    
    # Get top pages
    mom_top_pages = get_top_pages(mom_filtered, ['program_category', 'default_channel'], 'Session', 3)
    yoy_top_pages = get_top_pages(yoy_filtered, ['program_category', 'default_channel'], 'Session', 3)
    
    # Summarize data
    summary_data = summarize_data(mom_top_pages, yoy_top_pages)
    
    # Generate markdown report
    generate_markdown(summary_data, "landing_page_top3_next_month_v5_update.md")
