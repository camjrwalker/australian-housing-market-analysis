# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the data
df = pd.read_csv('D:/Users/CameronWalker/Documents/15. Work/New Roles - Data Science/Projects/Housing/raw_data/Australian_Housing_Affordability_2000_2025.csv')

# 2. Set Visual Style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1b2631'
plt.rcParams['axes.facecolor'] = '#1b2631'
plt.rcParams['grid.color'] = '#2c3e50'

# Create a 2x2 grid for the dashboard
fig, axes = plt.subplots(2, 2, figsize=(20, 12))
plt.subplots_adjust(hspace=0.3, wspace=0.2)
fig.suptitle('Australian Property Market Dynamics 2000-2025: Key Insights & Trends', fontsize=22, fontweight='bold', y=0.96)

# --- CHART 1: The Affordability Gap (Line Chart with 4 Y-Axes) ---
price_df = df[df['Metric'] == 'Median_House_Price'].set_index('Year')
income_df = df[df['Metric'] == 'Median_Household_Income'].set_index('Year')
mortgage_df = df[df['Metric'] == 'Avg_Mortgage_Rate_Percent'].set_index('Year')
students_df = df[df['Metric'] == 'International_Student_Numbers'].set_index('Year')

ax1 = axes[0, 0]

# Create three additional y-axes
ax1_income = ax1.twinx()  # Right side - Income
ax1_mortgage = ax1.twinx()  # Far left - Mortgage Rate
ax1_students = ax1.twinx()  # Far right - Student Numbers

# Offset the additional spines
ax1_mortgage.spines['left'].set_position(('axes', -0.15))  # Move left spine outward
ax1_mortgage.spines['left'].set_visible(True)
ax1_mortgage.yaxis.set_label_position('left')
ax1_mortgage.yaxis.set_ticks_position('left')

ax1_students.spines['right'].set_position(('axes', 1.15))  # Move right spine outward
ax1_students.spines['right'].set_visible(True)

# Define colors for each metric
color_price = '#f1c40f'      # Gold - House Prices
color_income = '#1abc9c'     # Teal - Income
color_mortgage = '#e74c3c'   # Red - Mortgage Rate
color_students = '#00FFFF'   # Cyan - Student Numbers

# Plot data
line1, = ax1.plot(price_df.index, price_df['National'] * 1000, color=color_price, marker='o', 
                   label='National Median Price', linewidth=2.5, markersize=4)
line2, = ax1_income.plot(income_df.index, income_df['National'] * 1000, color=color_income, 
                          linestyle='--', label='National Median Income', linewidth=3.0)
line3, = ax1_mortgage.plot(mortgage_df.index, mortgage_df['National'], color=color_mortgage, 
                            linestyle='-.', label='Mortgage Rate %', linewidth=2.5)
line4, = ax1_students.plot(students_df.index, students_df['National'], color=color_students, 
                            linestyle=':', label='Overseas Student Visas', linewidth=2.5)
'''
# Set axis labels with matching colors
ax1.set_ylabel('Median Price ($)', fontsize=10, color=color_price)
ax1_income.set_ylabel('Median Income ($ p.a.)', fontsize=10, color=color_income)
ax1_mortgage.set_ylabel('Mortgage Rate (%)', fontsize=10, color=color_mortgage)
ax1_students.set_ylabel('Overseas Student Visas', fontsize=10, color=color_students)

# Color the tick labels to match
ax1.tick_params(axis='y', colors=color_price)
ax1_income.tick_params(axis='y', colors=color_income)
ax1_mortgage.tick_params(axis='y', colors=color_mortgage)
ax1_students.tick_params(axis='y', colors=color_students)

# Color the spines to match
ax1.spines['left'].set_color(color_price)
ax1_income.spines['right'].set_color(color_income)
ax1_mortgage.spines['left'].set_color(color_mortgage)
ax1_students.spines['right'].set_color(color_students)
'''
# Set axis labels with matching colors
ax1.set_ylabel('Median Price ($)', fontsize=10, color='white')
ax1_income.set_ylabel('Median Income ($ p.a.)', fontsize=10, color='white')
ax1_mortgage.set_ylabel('Mortgage Rate (%)', fontsize=10, color='white')
ax1_students.set_ylabel('Overseas Student Visas', fontsize=10, color='white')

# Color the tick labels to match
ax1.tick_params(axis='y', colors='white')
ax1_income.tick_params(axis='y', colors='white')
ax1_mortgage.tick_params(axis='y', colors='white')
ax1_students.tick_params(axis='y', colors='white')

# Color the spines to match
ax1.spines['left'].set_color('white')
ax1_income.spines['right'].set_color('white')
ax1_mortgage.spines['left'].set_color('white')
ax1_students.spines['right'].set_color('white')

# Format y-axis labels with commas for thousands
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000000:.1f}M' if x >= 1000000 else f'${x/1000:.0f}K'))
ax1_income.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax1_mortgage.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
ax1_students.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))

# Set title
ax1.set_title('Housing Market Drivers: Incomes, Mortgage Rates &\nOS Student Visas (a leading indicator?)', fontsize=12, fontweight='bold')

# Create combined legend
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=8, framealpha=0.9)

# Adjust subplot to make room for outer axes
fig.subplots_adjust(left=0.12, right=0.88)

# --- CHART 2: City Affordability Heatmap (Heatmap) ---
heatmap_data = df[df['Metric'] == 'Price_to_Income_Ratio'].copy()
heatmap_data = heatmap_data.drop(columns=['Metric']).set_index('Year')

# Select specific years
years_to_show = [2000, 2005, 2010, 2015, 2020, 2025]
heatmap_data = heatmap_data.loc[heatmap_data.index.isin(years_to_show)]
heatmap_data = heatmap_data.sort_index()

# Reorder columns
city_order = ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Hobart', 'Darwin', 'Canberra', 'National']
heatmap_data = heatmap_data[[c for c in city_order if c in heatmap_data.columns]]

# Force numeric conversion
for col in heatmap_data.columns:
    heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')

ax2 = axes[0, 1]

# Create the heatmap WITHOUT seaborn annotations (they seem buggy)
hm = sns.heatmap(heatmap_data.astype(float), 
            annot=False,  # Disable seaborn annotations
            cmap='coolwarm', 
            ax=ax2, 
            linewidths=1,
            linecolor='#1b2631',
            cbar_kws={'label': 'Price-to-Income Ratio'})

# Manually add text annotations to each cell using matplotlib
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]
        # Choose text color based on cell value (dark text for light cells, light text for dark cells)
        text_color = 'white' if value > 7 else 'black'
        ax2.text(j + 0.5, i + 0.5, f'{value:.1f}', 
                ha='center', va='center', 
                fontsize=10, fontweight='bold', 
                color=text_color)

ax2.set_title('City Affordability Heatmap: Price-to-Income Ratio', fontsize=14)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)


# --- CHART 3: Market Velocity (Bar Chart) with Price vs Income Overlay ---
dom_df = df[df['Metric'] == 'Days_on_Market_Median']
dom_2000 = dom_df[dom_df['Year'] == 2000].drop(columns=['Year', 'Metric']).iloc[0]
dom_2025 = dom_df[dom_df['Year'] == 2025].drop(columns=['Year', 'Metric']).iloc[0]

ax3 = axes[1, 0]
x = np.arange(len(dom_2000))
width = 0.35

rects1 = ax3.bar(x - width/2, dom_2000, width, label='Days on Market 2000', color='#aed6f1')
rects2 = ax3.bar(x + width/2, dom_2025, width, label='Days on Market 2025', color='#2e86c1')

ax3.set_title('Market Velocity: Days on Market (2000 vs. 2025)\nwith National Price vs. Income Gap', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(dom_2000.index, rotation=45, ha='right')
# ax3.set_ylabel('Days on Market', color='#2e86c1')
# ax3.tick_params(axis='y', colors='#2e86c1')
ax3.set_ylabel('Days on Market', color='white')
ax3.tick_params(axis='y', colors='white')
ax3.bar_label(rects1, padding=3, fontsize=8)
ax3.bar_label(rects2, padding=3, fontsize=8)

# Create secondary Y-axis for Price and Income lines
ax3_twin = ax3.twinx()

# Get price and income data for line overlay (use year as proxy for x position)
price_national = df[df['Metric'] == 'Median_House_Price'].set_index('Year')['National'] * 1000
income_national = df[df['Metric'] == 'Median_Household_Income'].set_index('Year')['National'] * 1000

# Map years to x positions (spread across the bar chart width)
years = price_national.index.values
x_line = np.linspace(0, len(dom_2000) - 1, len(years))

# Plot price and income lines
line_price, = ax3_twin.plot(x_line, price_national.values, color='#f1c40f', marker='o', 
                             label='National Median Price', linewidth=2.5, markersize=3)
line_income, = ax3_twin.plot(x_line, income_national.values, color='#1abc9c', linestyle='--', 
                              label='National Median Income', linewidth=2.5)

ax3_twin.set_ylabel('Price / Income ($)', fontsize=10)
ax3_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Annotation for the widening gap
ax3_twin.annotate('Widening Gap:\nPrices outpace\nIncomes', xy=(7.5, 1000000), xytext=(4, 550000), 
                  arrowprops=dict(facecolor='grey', shrink=0.05), fontsize=10, ha='center', fontweight='bold')

# Combined legend
bars_labels = ['Days on Market 2000', 'Days on Market 2025']
lines_labels = ['National Median Price', 'National Median Income']
ax3.legend([rects1, rects2, line_price, line_income], bars_labels + lines_labels, 
           loc='upper left', fontsize=7, framealpha=0.9)


# --- CHART 4: Investment Returns (Stacked Area Chart) ---
yield_df = df[df['Metric'] == 'Rental_Yield_Percent'].set_index('Year').drop(columns=['Metric', 'National'])
ax4 = axes[1, 1]
ax4.stackplot(yield_df.index, yield_df.T, labels=yield_df.columns, alpha=0.8, colors=sns.color_palette("Spectral", 8))
ax4.set_title('Investment Returns: Average Rental Yield % Over Time', fontsize=14)
ax4.set_ylabel('Summed Rental Yield % pa')

# Move legend to bottom center, above the x-axis
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 0.18), fontsize=8, ncol=4, framealpha=0.9)

ax4.annotate('', xy=(2024, 30), xytext=(2002, 45), arrowprops=dict(facecolor='grey', shrink=0.05))

# Add yield values for 2000 and 2025
cities = yield_df.columns.tolist()
colors = sns.color_palette("Spectral", 8)

# Calculate cumulative positions for stacked area labels
cumsum_2000 = 0
cumsum_2025 = 0

for i, city in enumerate(cities):
    val_2000 = yield_df.loc[2000, city]
    val_2025 = yield_df.loc[2025, city]
    
    # Position labels at midpoint of each stack segment
    y_pos_2000 = cumsum_2000 + val_2000 / 2
    y_pos_2025 = cumsum_2025 + val_2025 / 2
    
    # Add 2000 labels (offset to the right)
    ax4.annotate(f'{val_2000:.1f}%', xy=(2000 + 1.5, y_pos_2000), fontsize=7, 
                 ha='center', va='center', fontweight='bold', color='black')
    
    # Add 2025 labels (offset to the left)
    ax4.annotate(f'{val_2025:.1f}%', xy=(2025 - 1.5, y_pos_2025), fontsize=7, 
                 ha='center', va='center', fontweight='bold', color='black')
    
    cumsum_2000 += val_2000
    cumsum_2025 += val_2025

plt.tight_layout(rect=[0.08, 0, 0.92, 0.95])
plt.show()