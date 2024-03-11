import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
credit = pd.read_csv('credit_record.csv')
application = pd.read_csv('application_record.csv')

# print(credit)
# print(application)

# print(len(set(application['ID'])))
# print(len(set(credit['ID'])))
# print(len(set(application['ID']).intersection(set(credit['ID']))))

grouped = credit.groupby('ID')

pivot_tb = credit.pivot(index='ID', columns='MONTHS_BALANCE', values='STATUS')
pivot_tb['open_month'] = grouped['MONTHS_BALANCE'].min()
pivot_tb['end_month'] = grouped['MONTHS_BALANCE'].max()
pivot_tb['ID'] = pivot_tb.index
pivot_tb = pivot_tb[['ID', 'open_month', 'end_month']]
pivot_tb['window'] = pivot_tb['end_month'] - pivot_tb['open_month']
pivot_tb.reset_index(drop=True, inplace=True)
credit = pd.merge(credit, pivot_tb, on='ID', how='left')
credit0 = credit.copy()
credit = credit[credit['window'] > 20]
credit['status'] = np.where((credit['STATUS'] == '2') | (credit['STATUS'] == '3') | (credit['STATUS'] == '4') | (credit['STATUS'] == '5'), 1, 0)
credit['status'] = credit['status'].astype(np.int8)
credit['month_on_book'] = credit['MONTHS_BALANCE'] - credit['open_month']
credit.sort_values(by=['ID', 'month_on_book'], inplace=True)

denominator = pivot_tb.groupby(['open_month']).agg({'ID': ['count']})
denominator.reset_index(inplace=True)
denominator.columns = ['open_month', 'sta_sum']

vintage = credit.groupby(['open_month', 'month_on_book']).agg({'ID': ['count']})
vintage.reset_index(inplace=True)
vintage.columns = ['open_month', 'month_on_book', 'sta_sum']
vintage['due_count'] = np.nan
vintage = vintage[['open_month', 'month_on_book', 'due_count']]
vintage = pd.merge(vintage, denominator, on=['open_month'], how='left')

for j in range(-60, 1):
    ls = []
    for i in range(0, 61):
        due = list(credit[(credit['status'] == 1) & (credit['month_on_book'] == i) & (credit['open_month'] == j)]['ID'])
        ls.extend(due)
        vintage.loc[(vintage['month_on_book'] == i) & (vintage['open_month'] == j), 'due_count'] = len(set(ls))

vintage['sta_rate'] = vintage['due_count'] / vintage['sta_sum']

vintage_wide = vintage.pivot(index='open_month', columns='month_on_book', values='sta_rate')

# plt.rcParams['figure.facecolor'] = 'white'
# vintage0 = vintage_wide.replace(0,np.nan)
# lst = [i for i in range(0, 61)]
# vintage_wide[lst].T.plot(legend=False, grid=True, title='Cumulative % of Bad Customers (> 60 Days Past Due)')
# plt.xlabel('Months on Book')
# plt.ylabel('Cumulative % > 60 Days Past Due')
# plt.savefig('vintage_analysis.png')

# lst = []
# for i in range(0, 61):
#     ratio = len(pivot_tb[pivot_tb['window'] < i]) / len(set(pivot_tb['ID']))
#     lst.append(ratio)
# pd.Series(lst).plot(legend=False, grid=True, title=' ')
# plt.xlabel('Observe Window')
# plt.ylabel('account ratio')
# plt.savefig('account_ratio.png')

def calculate_observe(credit, command):
    id_sum = len(set(pivot_tb['ID']))
    credit['status'] = 0
    exec(command)
    credit['month_on_book'] = credit['MONTHS_BALANCE'] - credit['open_month']
    minagg = credit[credit['status'] == 1].groupby('ID')['month_on_book'].min()
    minagg = pd.DataFrame(minagg)
    minagg['ID'] = minagg.index
    obslst = pd.DataFrame({'month_on_book':range(0,61), 'rate': None})
    lst = []
    for i in range(0,61):
        due = list(minagg[minagg['month_on_book']  == i]['ID'])
        lst.extend(due)
        obslst.loc[obslst['month_on_book'] == i, 'rate'] = len(set(lst)) / id_sum 
    return obslst['rate']

command = "credit.loc[(credit['STATUS'] == '0') | (credit['STATUS'] == '1') | (credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"   
morethan1 = calculate_observe(credit, command)
command = "credit.loc[(credit['STATUS'] == '1') | (credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"   
morethan30 = calculate_observe(credit, command)
command = "credit.loc[(credit['STATUS'] == '2') | (credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"
morethan60 = calculate_observe(credit, command)
command = "credit.loc[(credit['STATUS'] == '3' )| (credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"
morethan90 = calculate_observe(credit, command)
command = "credit.loc[(credit['STATUS'] == '4' )| (credit['STATUS'] == '5'), 'status'] = 1"
morethan120 = calculate_observe(credit, command)
command = "credit.loc[(credit['STATUS'] == '5'), 'status'] = 1"
morethan150 = calculate_observe(credit, command)

obslst = pd.DataFrame({'past due more than 30 days': morethan30,
                       'past due more than 60 days': morethan60,
                       'past due more than 90 days': morethan90,
                       'past due more than 120 days': morethan120,
                       'past due more than 150 days': morethan150
                        })

obslst.plot(grid = True, title = 'Cumulative % of Bad Customers Analysis')
plt.xlabel('Months on Books')
plt.ylabel('Cumulative %')
plt.savefig('obslst.png')

