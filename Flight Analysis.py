'''/****************************************************************************************************************************************************************
  Author:
  Dhairya Dhondiyal

  Description:
  A bot created on python selenium, that mines data on cheapest flights using google flights API.

  Written by Dhairya Dhondiyal, March 2017
****************************************************************************************************************************************************************/'''
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import scipy.spatial.distance
import re
import time
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import unicodedata
from datetime import datetime,timedelta
import pandas as pd
from dateutil.parser import parse
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
def task_4_dbscan(flight_data):
    flight_data['day'] = flight_data['Date_of_Flight'].dt.day
    X = StandardScaler().fit_transform(flight_data[['day', 'Price']])
    db = DBSCAN(eps=.404, min_samples=3).fit(X)
    labels = db.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    plt.subplots(figsize=(12, 8))
    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)
    flight_data['dbscan_labels'] = db.labels_
    unique_labels=[]
    for index,rows in flight_data.groupby('dbscan_labels').count().iterrows():
        if rows['Price']>=5:
            unique_labels.append(index)
    list_of_5_same_label=[]
    for label in unique_labels:
        count=0
        for index,rows in flight_data.iterrows():
            if rows['dbscan_labels']!=label:
                count = 0
                continue
            count+=1
            if count==5:
                list_of_5_same_label.append(rows['Date_of_Flight'])
                count=0
    for i in list_of_5_same_label:
        flag=1
        for index,rows in flight_data.loc[flight_data['Date_of_Flight'].isin([i,i-timedelta(days=1),i-timedelta(days=2),i-timedelta(days=3),i-timedelta(days=4)])].iterrows():
            for ind, row in flight_data.loc[flight_data['Date_of_Flight'].isin([i, i - timedelta(days=1), i - timedelta(days=2), i - timedelta(days=3), i - timedelta(days=4)])].iterrows():
                if math.fabs(rows['Price']-row['Price'])>20:
                    flag=0
                    list_of_5_same_label.remove(i)
                    break
            if flag==0:
                break
    min=float('inf')
    for i in list_of_5_same_label:
        temp=flight_data.loc[flight_data['Date_of_Flight'].isin([i, i - timedelta(days=1), i - timedelta(days=2), i - timedelta(days=3), i - timedelta(days=4)])]
        if temp[['Price']].mean()[0].item()<min:
            min=temp[['Price']].mean().item()
            date=i
    clean_data=[]
    for index,rows in flight_data.loc[flight_data['Date_of_Flight'].isin([date, date - timedelta(days=1), date - timedelta(days=2), date - timedelta(days=3), date - timedelta(days=4)])].iterrows():
        clean_data.append([rows['Price'], rows['Date_of_Flight']])
    return pd.DataFrame(clean_data,columns=['Price','Date_of_Flight'])

def task_3_IQR(flight_data):
    plot=plt.boxplot(flight_data['Price'],patch_artist=True)
    for median in plot['medians']:
        median.set(color='#fc0004', linewidth=2)
    for flier in plot['fliers']:
        flier.set(marker='+', color='#e7298a')
    for whisker in plot['whiskers']:
        whisker.set(color='#7570b3', linewidth=2)
    for cap in plot['caps']:
        cap.set(color='#7570b3', linewidth=2)
    for box in plot['boxes']:
        box.set(color='#7570b3', linewidth=2)
        box.set(facecolor='#1b9e77')
    plt.matplotlib.pyplot.savefig('task_3_iqr.png')
    clean_data=[]
    for index,row in flight_data.loc[flight_data['Price'].isin(plot['fliers'][0].get_ydata())].iterrows():
        clean_data.append([row['Price'],row['Date_of_Flight']])
    return pd.DataFrame(clean_data, columns=['Price', 'Date_of_Flight'])

def task_3_dbscan(flight_data):
    flight_data['day'] = flight_data['Date_of_Flight'].dt.day
    X = StandardScaler().fit_transform(flight_data[['day','Price']])
    db = DBSCAN(eps=.404, min_samples=3).fit(X)
    labels = db.labels_
    clusters = len(set(labels))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    plt.subplots(figsize=(12, 8))
    for k, c in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=c,
                 markeredgecolor='k', markersize=14)

    plt.title("Total Clusters: {}".format(clusters), fontsize=14, y=1.01)
    flight_data['dbscan_labels'] = db.labels_
    plt.matplotlib.pyplot.savefig('task_3_dbscan.png')
    plt.clf()
    outliers=flight_data.loc[flight_data['dbscan_labels']==-1]
    clean_data = []
    for Index,Row in outliers.iterrows():
        minimum = float('inf')
        for label in unique_labels:
            if label==-1:
                continue
            t=flight_data.loc[flight_data['dbscan_labels']==label]
            t=t.as_matrix(columns=['Price','day'])
            t=t.mean(axis=0)
            if math.sqrt((Row['Price']-t[0])**2+(Row['day']-t[1])**2)<minimum:
                minimum=math.sqrt((Row['Price']-t[0])**2+(Row['day']-t[1])**2)
                closest_cluster=label
                closest_cluster_price=t[0]
        if(closest_cluster_price>Row['Price']):
            min = float('inf')
            counter=0
            sum=0
            for index,row in flight_data.iterrows():
                if row['dbscan_labels'] == closest_cluster:
                    sum=sum+((row['Price']-closest_cluster_price)**2)
                    counter+=1
            if Row['Price']<=max(closest_cluster_price-2*math.sqrt(sum/counter),50):
                clean_data.append([Row['Price'],Row['Date_of_Flight']])
    return pd.DataFrame(clean_data, columns=['Price', 'Date_of_Flight'])
def scrape_data(start_date,from_place,to_place,city_name):
    driver = webdriver.Chrome()
    driver.get('https://www.google.com/flights/explore/')
    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.LJTSM3-k-b")))
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]/div/div')
    from_input.click()
    actions = ActionChains(driver)
    actions.send_keys(from_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()
    actions = ActionChains(driver)
    actions.send_keys(to_place)
    time.sleep(0.1)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.LJTSM3-v-d")))
    manip_url=re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}",datetime.strftime(start_date,"%Y-%m-%d"),driver.current_url)
    driver.quit()
    driver = webdriver.Chrome()
    driver.get(manip_url)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.LJTSM3-v-d")))
    data=[]
    results = driver.find_elements_by_class_name('LJTSM3-v-d')
    for result in results:
        if ''.join(c for c in unicodedata.normalize('NFD',result.find_elements_by_class_name('LJTSM3-v-c')[0].text)if unicodedata.category(c) != 'Mn').lower().find(city_name.lower())!=-1:
            bars = result.find_elements_by_class_name('LJTSM3-w-x')
            for bar in bars:
                ActionChains(driver).move_to_element(bar).perform()
                time.sleep(0.01)
                data.append((result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))
    if not data:
        print "Data Scraping failed"
        exit()
    clean_data=[]
    for d in data:
        clean_data.append([float(d[0].replace('$', '').replace(',','')),(parse(d[1].split('-')[0].strip()))])
    df = pd.DataFrame(clean_data,columns=['Price','Date_of_Flight'])
    driver.quit()
    return df
def scrape_data_90(start_date,from_place,to_place,city_name):
    driver = webdriver.Chrome()
    driver.get('https://www.google.com/flights/explore/')
    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.LJTSM3-k-b")))
    from_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[2]/div/div')
    from_input.click()
    actions = ActionChains(driver)
    actions.send_keys(from_place)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    to_input = driver.find_element_by_xpath('//*[@id="root"]/div[3]/div[3]/div/div[4]/div/div')
    to_input.click()
    actions = ActionChains(driver)
    actions.send_keys(to_place)
    time.sleep(0.1)
    actions.send_keys(Keys.ENTER)
    actions.perform()
    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.LJTSM3-v-d")))
    manip_url=re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}",datetime.strftime(start_date,"%Y-%m-%d"),driver.current_url)
    driver.quit()
    driver = webdriver.Chrome()
    driver.get(manip_url)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.LJTSM3-v-d")))
    counter=0
    data=[]
    results = driver.find_elements_by_class_name('LJTSM3-v-d')
    for result in results:
        if ''.join(c for c in unicodedata.normalize('NFD',result.find_elements_by_class_name('LJTSM3-v-c')[0].text)if unicodedata.category(c) != 'Mn').lower().find(city_name.lower())!=-1:
            bars = result.find_elements_by_class_name('LJTSM3-w-x')
            for bar in bars:
                counter+=1
                ActionChains(driver).move_to_element(bar).perform()
                time.sleep(0.01)
                data.append((result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))
    if not data:
        print "Data Scraping failed"
        exit()
    clean_data=[]
    for d in data:
        clean_data.append([float(d[0].replace('$', '').replace(',','')),(parse(d[1].split('-')[0].strip()))])
    driver.quit()
    manip_url = re.sub("[0-9]{4}-[0-9]{2}-[0-9]{2}", datetime.strftime(start_date+timedelta(days=60), "%Y-%m-%d"), manip_url)
    driver = webdriver.Chrome()
    driver.get(manip_url)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.LJTSM3-v-d")))
    data = []
    results = driver.find_elements_by_class_name('LJTSM3-v-d')
    for result in results:
        if ''.join(c for c in unicodedata.normalize('NFD', result.find_elements_by_class_name('LJTSM3-v-c')[0].text) if
                   unicodedata.category(c) != 'Mn').lower().find(city_name.lower()) != -1:
            bars = result.find_elements_by_class_name('LJTSM3-w-x')
            for bar in bars:
                if counter==90:
                    break
                counter += 1
                ActionChains(driver).move_to_element(bar).perform()
                time.sleep(0.01)
                data.append((result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[0].text,result.find_element_by_class_name('LJTSM3-w-k').find_elements_by_tag_name('div')[1].text))
            if counter==90:
                break
    for d in data:
        clean_data.append([float(d[0].replace('$', '').replace(',','')), (parse(d[1].split('-')[0].strip()))])
    df = pd.DataFrame(clean_data, columns=['Price', 'Date_of_Flight'])
    driver.quit()
    return df

#date=datetime.strptime("2017-04-17","%Y-%m-%d")
#df=scrape_data(date,"New York","Russia","Moscow")
#print task_3_dbscan(df)
#task_3_IQR(df)
#task_4_dbscan(df)
#scrape_data_90(date,"EWR","Mexico","Cancun")
