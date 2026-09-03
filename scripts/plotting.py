import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap, Normalize
import contextily as ctx
import numpy as np
from pyproj import Transformer
import cmcrameri.cm as cmc
import rasterio
import seaborn as sns

lcz = {
  1: {"color": "#8c0000", "label": "Compact High-Rise", "lcz_class": "1", "alpha": 255},
  2: {"color": "#d10000", "label": "Compact Mid-Rise", "lcz_class": "2", "alpha": 255},
  3: {"color": "#ff0000", "label": "Compact Low-Rise", "lcz_class": "3", "alpha": 255},
  4: {"color": "#bf4d00", "label": "Open High-Rise", "lcz_class": "4", "alpha": 255},
  5: {"color": "#ff6600", "label": "Open Mid-Rise", "lcz_class": "5", "alpha": 255},
  6: {"color": "#ff9955", "label": "Open Low-Rise", "lcz_class": "6", "alpha": 255},
  7: {"color": "#faee05", "label": "Lightweight low-rise", "lcz_class": "7", "alpha": 255},
  8: {"color": "#bcbcbc", "label": "Large low-rise", "lcz_class": "8", "alpha": 255},
  9: {"color": "#ffccaa", "label": "Sparsely built", "lcz_class": "9", "alpha": 255},
  10: {"color": "#555555", "label": "Heavy industry", "lcz_class": "10", "alpha": 255},
  11: {"color": "#006a00", "label": "Dense trees", "lcz_class": "A", "alpha": 255},
  12: {"color": "#00aa00", "label": "Scattered trees", "lcz_class": "B", "alpha": 255},
  13: {"color": "#648525", "label": "Bush or scrub", "lcz_class": "C", "alpha": 255},
  14: {"color": "#b9db79", "label": "Low plants", "lcz_class": "D", "alpha": 255},
  15: {"color": "#000000", "label": "Bare rock or paved", "lcz_class": "E", "alpha": 255},
  16: {"color": "#fbf7ae", "label": "Bare soil or sand", "lcz_class": "F", "alpha": 255},
  17: {"color": "#6a6aff", "label": "Water", "lcz_class": "G", "alpha": 255}
}



def obs_map(data: pd.DataFrame, value_col: str, api_key:bool=False):
    """Create a map of the observation locations and a single parameter

    Args:
        data (pd.DataFrame): dataframe with lat/lon columns and observational data
        value_col (str): one of "tmean", "uhi_mean" and "rmse"
        api_key (bool, optional): Indicator whether
            API key to CartoDB tile server installed. Defaults to False.

    """
    stn = data.copy()
    # we want to plot using a projection that is suited to all of Europe instead of Mercator
    # x and y represent easting and northing in this CRS, respectively
    to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    stn["x"], stn["y"] = to_3035.transform(stn["longitude"].values, stn["latitude"].values)


    # we set up the extent of the figure by adding a bit of padding around the extreme locations 
    # of the stations
    padx = (stn["x"].max() - stn["x"].min()) * 0.05
    pady = (stn["y"].max() - stn["y"].min()) * 0.05

    # we create the figure with the given dimensions
    fig, ax = plt.subplots(figsize=(20, 18), dpi=300)
    ax.set_xlim(stn["x"].min() - padx, stn["x"].max() + padx)
    ax.set_ylim(stn["y"].min() - pady, stn["y"].max() + pady)
    ax.set_aspect("equal")

    if value_col == "tmean":
        # scatter the stations on the map and color the dots according to their mean temperature
        sc = ax.scatter(stn["x"], stn["y"], c=stn[value_col],
                        cmap=cmc.roma_r, s=45, edgecolors="black")
        for (label, x, y) in zip ([l.split("_")[-1] for l in stn.index], stn["x"], stn["y"]):
            ax.annotate(label, xy=(x,y), textcoords='offset points', xytext=(5,0))
        # colorbar = legend
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Mean annual air temperature (°C)")

    elif value_col == "uhi_mean":
        # scatter the stations on the map and color the dots according to their mean temperature
        sc = ax.scatter(stn["x"], stn["y"], c=stn[value_col],
                        cmap=cmc.vik, s=45, edgecolors="none")
        # colorbar = legend
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Mean UHI (°C)")

    elif value_col == "rmse":
        # shared color scale so square (train) and circle (val) colors are comparable
        vmin, vmax = stn["rmse"].min(), stn["rmse"].max()

        train = stn[stn["split"] == "train"]
        val   = stn[stn["split"] == "val"]

        sc = ax.scatter(train["x"], train["y"], c=train["rmse"], cmap=cmc.batlow,
                        vmin=vmin, vmax=vmax, marker="s", s=55,
                        edgecolors="black", linewidths=0.4, label="train")
        ax.scatter(val["x"], val["y"], c=val["rmse"], cmap=cmc.batlow,
                   vmin=vmin, vmax=vmax, marker="o", s=55,
                   edgecolors="black", linewidths=0.4, label="val")

        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Station RMSE (°C)")
        ax.legend(loc="upper right")

    # add a basemap to situate ourselves
    if not api_key:
        # use default basemap
        ctx.add_basemap(ax, crs="EPSG:3035")
    else:
        ctx.add_basemap(ax, crs="EPSG:3035",
                        source = ctx.providers.CartoDB.Positron)

    ax.set_xticks([]); ax.set_yticks([])

    plt.show()

    return fig, ax



def station_surroundings(station:str, lat:float, lon:float, api_key:bool=False):
    """Create a map of a single observation locations

    Args:
        station(str): name of the station
        lat(float): latitude of the station
        lon(float): longitude of the station
        api_key (bool, optional): Indicator whether
            API key to CartoDB tile server installed. Defaults to False.

    """
    to_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = to_3035.transform(lon, lat)
    df = pd.DataFrame({"Latitude":[lat], "Longitude":[lon]})
    gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
            )

    gdf = gdf.to_crs(3035)  # Convert the dataset to a coordinate
    # system which uses meters



    # we set up the extent of the figure by adding a bit of padding around the extreme locations
    # of the stations
    padx = x * 0.0005
    pady = y * 0.0005

    # we create the figure with the given dimensions
    fig, ax = plt.subplots(dpi=140)
    ax.set_xlim(x - padx, x + padx)
    ax.set_ylim(y - pady, y + pady)
    ax.set_aspect("equal")

    ax.plot(x, y, marker=".", color="black")

    # add a basemap to situate ourselves
    if not api_key:
        # use default basemap
        ctx.add_basemap(ax, crs="EPSG:3035")
    else:
        ctx.add_basemap(ax, crs="EPSG:3035",
                        source = ctx.providers.CartoDB.Positron)

    from matplotlib_scalebar.scalebar import ScaleBar
    ax.set_aspect(1)
    ax.add_artist(ScaleBar(1))
    ax.set_title(station)
    ax.set_xticks([]); ax.set_yticks([])

    plt.show()


def week_plot(hot_week:pd.DataFrame, cold_week:pd.DataFrame, hot_id:str, cold_id:str):
    """Create a twofold lineplot for the hottest week in the time period

    Args:
        hot_week (pd.DataFrame):  dataframe with a week of data
        cold_week (pd.DataFrame): dataframe with a week of data
        hot_id (str): label of warm station
        cold_id (str): label of cool station
    """

    c_hot = cmc.vik(0.8)   # warm end
    c_cold = cmc.vik(0.2)  # cool end

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hot_week["datetime_utc"], hot_week["air_temperature"],
            color=c_hot, label=f"{hot_id} (warmest mean)")
    ax.plot(cold_week["datetime_utc"], cold_week["air_temperature"],
            color=c_cold, label=f"{cold_id} (coolest mean)")

    ax.yaxis.set_major_locator(MultipleLocator(5))   # major ticks every 5 °C
    ax.yaxis.set_minor_locator(MultipleLocator(1))   # minor ticks every 1 °C
    ax.grid(which="major", axis="y", linewidth=0.8, alpha=0.6)
    ax.grid(which="minor", axis="y", linewidth=0.4, alpha=0.3)
    ax.set_title("Comparison of two stations in a heat wave, one with the hottest and one with the coldest data")
    ax.set_ylabel("Air temperature (°C)")
    ax.legend()
    fig.autofmt_xdate()      
    plt.show()
    return fig, ax


def quick_raster_plot(city: str, raster_type: str):
    """Create a city map with a certain raster as map

    Args:
        city (str): name of city, one of [bern, dortmund, freiburg, ghent]
        raster_type (str): name of raster type, [lcz, dtm]
    """
    if raster_type == "lcz":
        res = 100
    elif raster_type == "dtm":
        res = 30
    else:
        res = 10

    src = rasterio.open(f"../data/{raster_type}/{city}_{res}.tif")
    band = src.read(1)

    if raster_type == "lcz":
        cmap = ListedColormap([lcz_desc["color"] for lcz_desc in lcz.values])

    elif raster_type == "bh":
        # mask knows which data has the noData value
        mask = src.read_masks(1)
        # so we set the noData value to 0 because there are no buildings there
        band[mask==0] = 0
        cmap = cmc.grayC_r

    elif raster_type == "dtm":
        cmap = ListedColormap(cmc.bukavu(np.linspace(0.5, 1.0, 256)))

    elif raster_type == "tcd":
        cmap = ListedColormap(cmc.bam(np.linspace(0.5, 1.0, 256)))

    norm = Normalize(vmin=np.nanmin(band), vmax=np.nanmax(band))
    # plt.imshow(band, cmap=lcm, norm=norm)
    plt.imshow(band, cmap=cmap, norm=norm if raster_type in ("dtm", "tcd") else None)
    plt.colorbar()
    plt.show()


def plot_lcz_based_observations(df:pd.DataFrame, city:str):
    """Create a boxplot to categorize air temperature readings into lcz

    Args:
        df (pd.DataFrame): dataframe with air_temperature and lcz columns
        city (str): name of the city
    """
    # determine lcz classes in dataframe
    covered_lcz_classes = [c for c in df["lcz_nearest"].unique()]
    covered_lcz_classes.sort()

    # get color codes for each contained class
    box_colors = [lcz[id]["color"] for id in covered_lcz_classes]

    # plot all 'air_temperature' readings by lcz
    plt.figure(dpi=300)
    ax = df.boxplot(column="air_temperature", by="lcz_nearest", patch_artist=True, figsize=(18,6))
    ax.set_title(f"air_temperature in {city}")
    ax.set_xticklabels([lcz[id]["lcz_class"] for id in covered_lcz_classes])

    # fill with colors and add measurement count per class
    for i, (patch, color, count) in enumerate(zip(ax.patches, box_colors, df.groupby("lcz_nearest")["station_id"].count())):
        patch.set_facecolor(color)
        y_pos = 0

        ax.text(
            i+1,   # at box
            y_pos, # at air_temperature = 0
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 1, "edgecolor": "none", "boxstyle": "round,pad=0.3"}
        )

    plt.tight_layout()
    plt.show()


def plot_diurnal_cycle(df:pd.DataFrame, param:str, city:str, station:str | None=None):
    """Create a lineplot with confidence intervals

    Args:
        df (pd.DataFrane): dataset of observations of a city
        param (str): parameter to plot
        city (str): name of the city
        station (str, optional): name of a certain station. Defaults to None.
    """
    sns.set_style("whitegrid")
    sns.lineplot(
        data=df,
        x="hour",
        y=param,
        errorbar=("sd", 2),  # Standard deviation with 95% CI (2*sd)
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=8,
        markeredgewidth=1.5,
        markeredgecolor="white",
    )

    # Customize
    if station is not None:
        station_info = f"at station {station} "
    else:
        station_info=""
    plt.title(f"Mean diurnal cycle of {param} {station_info}in {city}", fontsize=14, pad=20)
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel(f"{param} (°C)", fontsize=12)
    plt.xticks(range(0, 24))
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_diurnal_cycle_city_station(df:pd.DataFrame, param:str, city:str, station:str | None=None):
    """Create a lineplot with confidence intervals for city and station

    Args:
        df (pd.DataFrame): dataset of observations of a city
        param (str): parameter to plot
        city (str): name of the city
        station (str, optional): name of a certain station. Defaults to None.
    """
    _, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")

    # plot for the city
    sns.lineplot(
        data=df,
        x="hour",
        y=param,
        # errorbar=("sd", 2),  # Standard deviation with 95% CI (2*sd)
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=8,
        markeredgewidth=1.5,
        markeredgecolor="white",
        ax=ax,
        label=f"{city} average"
    )

    # plot for the station subset
    sns.lineplot(
        data=df[df["station_id"]==station],
        x="hour",
        y=param,
        # errorbar=("sd", 2),  # Standard deviation with 95% CI (2*sd)
        color="green",
        linewidth=2.5,
        marker="o",
        markersize=8,
        markeredgewidth=1.5,
        markeredgecolor="white",
        ax=ax,
        label=station
    )

    # Customize
    plt.title(f"Mean diurnal cycle of {param} in {city} and at {station}", fontsize=14, pad=20)
    plt.xlabel("Hour of Day", fontsize=12)
    plt.ylabel(f"{param} (°C)", fontsize=12)
    plt.xticks(range(0, 24))
    sns.despine()
    plt.tight_layout()
    plt.show()
