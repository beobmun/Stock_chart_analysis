import pandas as pd
import numpy as np
import random
import torch
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PIL import Image, ImageChops
import torchvision.transforms as transforms
from tqdm import tqdm

class DataInfo:
    def __init__(self, info_path, data_path, random_state=42):
        self.info_path = info_path
        self.data_path = data_path
        self.random_state = random_state
        self.snp500_info_df = pd.DataFrame()
        self.snp500_symbols = list()
        self.train_symbols = list()
        self.val_symbols = list()
        self.test_symbpls = list()
        
    def load_info(self, listed_date='2015-01-01'):
        listed_date = pd.to_datetime(listed_date)
        try:
            self.snp500_info_df = pd.read_csv(self.info_path)
            self.snp500_info_df.loc[:, 'Date added'] = pd.to_datetime(self.snp500_info_df['Date added'], format='%Y-%m-%d')
            self.snp500_info_df = self.snp500_info_df.loc[self.snp500_info_df['Date added'] <= listed_date]
            self.snp500_symbols = self.snp500_info_df['Symbol'].tolist()
        except FileNotFoundError:
            print(f"File not found: {self.info_path}")
        return self
    
    def train_test_split(self, train_size=0.6, val=True, sector=True):
        if not self.snp500_symbols:
            print("S&P 500 symbols are empty. Please load S&P 500 info first.")
            return self
        if sector:
            self.train_symbols = set(self.snp500_info_df.groupby('GICS Sector').apply(
                lambda x: x.sample(frac=train_size, random_state=self.random_state)
            )['Symbol'].tolist())
            if val:
                temp_df = self.snp500_info_df[~self.snp500_info_df['Symbol'].isin(self.train_symbols)]
                self.val_symbols = set(temp_df.groupby('GICS Sector').apply(
                    lambda x: x.sample(frac=0.5, random_state=self.random_state)
                )['Symbol'].tolist())
                self.test_symbols = set(temp_df['Symbol'].tolist()) - self.val_symbols
            else:
                self.test_symbols = set(self.snp500_symbols) - self.train_symbols
        else:
            self.train_symbols = set(self.snp500_info_df.sample(frac=train_size, random_state=self.random_state)['Symbol'].tolist())
            if val:
                temp_df = self.snp500_info_df[~self.snp500_info_df['Symbol'].isin(self.train_symbols)]
                self.val_symbols = set(temp_df.sample(frac=0.5, random_state=self.random_state)['Symbol'].tolist())
                self.test_symbols = set(temp_df['Symbol'].tolist()) - self.val_symbols
            else:
                self.test_symbols = set(self.snp500_symbols) - self.train_symbols
        
        self.train_symbols = list(self.train_symbols)
        self.val_symbols = list(self.val_symbols)
        self.test_symbols = list(self.test_symbols)
        return self
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df_cache, num_days=20, start_date=None, end_date=None, transform=None):
        self.df_cache = df_cache
        self.symbols = list(df_cache.keys())
        self.num_days = num_days
        self.start_date = start_date
        self.end_date = end_date
        self.transform = transform
        
    def __len__(self):
        return len(self.df_cache)
    
    def _trim(self, img):
        bg = Image.new(img.mode, img.size, (255, 255, 255))
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        return img.crop(bbox) if bbox else img
    
    def _generate_candle_chart(self, df, start):
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
        fig, _ = mpf.plot(df[start:start+self.num_days], type='candle', style=style, volume=True, ylabel='', ylabel_lower='', returnfig=True, figsize=(2.24, 2.24), volume_exponent=0)
        canvas = FigureCanvas(fig)
        canvas.draw()
        
        buf = BytesIO()
        # fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        fig.savefig(buf, format='png', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        if self.transform is not None:
            return self.transform(self._trim(Image.open(buf).convert('RGB')))
        else:
            return transforms.ToTensor()(self._trim(Image.open(buf).convert('RGB')))

    def _calc_yield(self, df, idx):
        yield_rate = (df.iloc[idx+self.num_days]['close'] - df.iloc[idx+self.num_days-1]['close']) / df.iloc[idx+self.num_days-1]['close'] * 100
        return yield_rate
    
    def __getitem__(self, idx):
        filtered_df = self.df_cache[self.symbols[idx]].loc[self.start_date:self.end_date]
        sample_idx = random.randint(1, len(filtered_df)-self.num_days - 2)
        chart_imgs = [
            self._generate_candle_chart(filtered_df, sample_idx - 1),
            self._generate_candle_chart(filtered_df, sample_idx),
            self._generate_candle_chart(filtered_df, sample_idx + 1)
        ]
        chart_imgs = torch.stack(chart_imgs)
        yield_rate = self._calc_yield(filtered_df, sample_idx)
        return chart_imgs, yield_rate
    
class Dataset2(torch.utils.data.Dataset):
    def __init__(self, df_cache, num_days=20, start_date=None, end_date=None, data_dir="", transform=None):
        # self.df_cache = df_cache
        # self.symbols = list(df_cache.keys())
        self.num_days = num_days
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.transform = transform
        
        self.samples = list()
        with tqdm(total=len(df_cache), desc="Preparing dataset samples", ncols=100, leave=False) as pbar:
            for symbol, df in df_cache.items():
                s = symbol.lower().replace(".", "-")
                df = df.loc[start_date:end_date]
                dates = df.index[num_days-1:]
                
                for i in range(1, len(dates)-1):
                    try:
                        pre_date = dates[i-1]
                        cur_date = dates[i]
                        next_date = dates[i+1]
                        
                        yield_rate = (df.loc[next_date]['close'] - df.loc[cur_date]['close']) / df.loc[cur_date]['close'] * 100
                        
                        self.samples.append({
                            'symbol': s,
                            'dates': [str(pre_date).split(' ')[0], str(cur_date).split(' ')[0], str(next_date).split(' ')[0]],
                            'yield': yield_rate
                        })
                    except Exception as e:
                        print(f"Error processing {symbol} on dates {dates[i-1]}, {dates[i]}, {dates[i+1]}: {e}")
                        continue
                pbar.update(1)
            
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        symbol, dates, yield_rate = sample['symbol'], sample['dates'], sample['yield']
        chart_imgs = [Image.open(f"{self.data_dir}/{self.num_days}/{symbol}/{d}.png").convert('RGB') for d in dates]
        if self.transform is not None:
            chart_imgs = [self.transform(img) for img in chart_imgs]
        chart_imgs = torch.stack(chart_imgs)
        return chart_imgs, yield_rate
