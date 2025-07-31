import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageChops
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager

def load_info(info_path):
    try:
        snp500_info_df = pd.read_csv(info_path)
        snp500_info_df.loc[:, 'Date added'] = pd.to_datetime(snp500_info_df['Date added'], format='%Y-%m-%d')
        return snp500_info_df
    except FileNotFoundError:
        print(f"File not found: {info_path}")
        return pd.DataFrame()
    
def trim(img):
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    return img.crop(bbox) if bbox else img

def generate_candle_chart(df, start, num_days):
    up_color = "#137901"
    down_color = "#d51217"
    marketcolor = mpf.make_marketcolors(up=up_color, down=down_color, 
                                        edge={'up': up_color, 'down': down_color},
                                        wick={'up': up_color, 'down': down_color},
                                        volume={'up': up_color, 'down': down_color},
                                        vcdopcod=True)
    style = mpf.make_mpf_style(base_mpf_style="default",
                                marketcolors=marketcolor,
                                facecolor='white',
                                rc={'xtick.bottom': False, 'xtick.labelbottom': False,
                                    'ytick.left': False, 'ytick.labelleft': False,
                                    'axes.spines.top': False, 'axes.spines.right': False,
                                    'axes.spines.left': False, 'axes.spines.bottom': False,
                                    'axes.grid': False})
    filtered_df = df[start:start+num_days]
    date = str(filtered_df.index[-1]).split(' ')[0]
    fig, _ = mpf.plot(filtered_df, type='candle', style=style, volume=True, ylabel='', ylabel_lower='', returnfig=True, figsize=(2.24, 2.24), volume_exponent=0)
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    
    return trim(img), date

def get_stock_data(data_path, symbol, start_date='2015-01-01', end_date='2024-12-31'):
    df = pd.read_csv(f"{data_path}/{symbol}.us.txt")
    df = df.rename(columns={
        "<DATE>": "date",
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<VOL>": "volume",
        })
    df.loc[:, "data"] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('data', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    df = df.sort_index()
    df = df.loc[start_date:end_date]
    return df

def process_symbol(args):
    symbol, data_path, img_save_path, num_days, start_date, end_date = args
    try:
        symbol_std = symbol.lower().replace(".", "-")
        save_path = f"{img_save_path}/{num_days}/{symbol_std}"
        os.makedirs(save_path, exist_ok=True)
        
        df = get_stock_data(data_path, symbol_std, start_date, end_date)
        
        for i in range(len(df) - num_days + 1):
            img, date = generate_candle_chart(df, start=i, num_days=num_days)
            img.save(f"{save_path}/{date}.png")

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    info_path = '../data/info/snp500_info.csv'
    data_path = '../data/US/time_series'
    img_save_path = '../data/US/chart_image'
    start_date = '2015-01-01'
    end_date = '2024-12-31'
    num_days = 20
    info_df = load_info(info_path)
    symbols = info_df['Symbol'].tolist()
    
    task_args = [(symbol, data_path, img_save_path, num_days, start_date, end_date) for symbol in symbols]
    
    print(f"Using {cpu_count()} CPU cores for processing.")
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_symbol, task_args), total=len(task_args), desc="Generating charts", ncols=100))
    
    print("All charts generated successfully.")