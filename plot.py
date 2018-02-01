import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import csv
import numpy as np

def plot_map(category, lat, lon):
    m = Basemap(projection='merc',llcrnrlat=20,urcrnrlat=50,\
                llcrnrlon=-130,urcrnrlon=-60,lat_ts=20,resolution='i')
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    # draw parallels and meridians.
    parallels = np.arange(-90.,91.,5.)
    # Label the meridians and parallels
    m.drawparallels(parallels,labels=[True,False,False,False])
    # Draw Meridians and Labels
    meridians = np.arange(-180.,181.,10.)
    m.drawmeridians(meridians,labels=[True,False,False,True])
    m.drawmapboundary(fill_color='white')
    plt.title("News in different categories")
    # Define a colormap
    cm = plt.cm.get_cmap('RdYlBu')
    # Transform points into Map's projection
    x,y = m(lon, lat)
    # Color the transformed points
    sc = plt.scatter(x,y, c=category, vmin=0, vmax =4, cmap=cm, s=20, edgecolors='none')

    plt.show()

def main():
    file_name = "0_data.csv"
    input_csv = open(file_name, 'r')
    category = []
    lat = []
    lon = []
    reader = csv.DictReader(input_csv)

    for row in reader:
        t_a = row['latitude']
        t_o = row['longitude']
        if t_a and t_o:
            category.append(int(row['category']))
            lat.append(float(t_a))
            lon.append(float(t_o))

    plot_map(category, lat, lon)

if __name__ == "__main__":
    # tf.app.run()
    main()
