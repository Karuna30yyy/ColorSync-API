from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import requests
import json
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from dataclasses import dataclass
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import pymongo
from pymongo import MongoClient
from datetime import datetime, timedelta
import os

class MongoDBHandler:
    def __init__(self, mongodb_url: str):
        try:
            self.client = MongoClient(mongodb_url)
            self.db = self.client['color_sync_db']
            self.search_cache = self.db['search_cache']
            
            self.search_cache.create_index([
                ('keyword', pymongo.ASCENDING), 
                ('color', pymongo.ASCENDING), 
                ('timestamp', pymongo.DESCENDING)
            ])
            
            self.search_cache.create_index('timestamp', expireAfterSeconds=3600)
        
        except Exception as e:
            print(f"MongoDB Connection Error: {e}")
            raise
    
    def cache_search_results(self, keyword: str, color: str, results: List[Dict]):
        try:
            cache_entry = {
                'keyword': keyword,
                'color': color,
                'results': results,
                'timestamp': datetime.utcnow()
            }
            self.search_cache.insert_one(cache_entry)
        except Exception as e:
            print(f"Error caching search results: {e}")
    
    def get_cached_results(self, keyword: str, color: str) -> Optional[List[Dict]]:
        try:
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            cached_entry = self.search_cache.find_one({
                'keyword': keyword, 
                'color': color, 
                'timestamp': {'$gte': one_hour_ago}
            }, sort=[('timestamp', pymongo.DESCENDING)])
            
            return cached_entry['results'] if cached_entry else None
        
        except Exception as e:
            print(f"Error retrieving cached results: {e}")
            return None

MYNTRA_COLORS = {
    'e8e6cf': 'Beige', '36454f': 'Black', '0074D9': 'Blue', 'cc8240': 'Bronze',
    '8b4513': 'Brown', 'a03245': 'Burgundy', '4b302f': 'Coffee Brown',
    'aa6c39': 'Copper', 'ff7f50': 'Coral', 'ede6b9': 'Cream',
    '8dc04a': 'Fluorescent Green', 'e5c74a': 'Gold', '5eb160': 'Green',
    '9fa8ab': 'Grey Melange', '808080': 'Grey', 'c3b091': 'Khaki',
    'd6d6e5': 'Lavender', '5db653': 'Lime Green', 'b9529f': 'Magenta',
    'b03060': 'Maroon', 'e0b0ff': 'Mauve', 'e0d0c5': 'Metallic',
    'cc9c33': 'Mustard', '3c4477': 'Navy Blue', 'dbaf97': 'Nude',
    'f2f2f2': 'Off White', '3D9970': 'Olive', 'f28d20': 'Orange',
    'ffe5b4': 'Peach', 'f1a9c4': 'Pink', '800080': 'Purple',
    'd34b56': 'Red', 'dd2f86': 'Rose', 'b7410e': 'Rust',
    '2e8b57': 'Sea Green', 'b3b3b3': 'Silver', 'd2b48c': 'Tan',
    '483c32': 'Taupe', '008080': 'Teal', '40e0d0': 'Turquoise Blue',
    'eadc32': 'Yellow'
}

HM_COLORS = {
    'f5f5dc': 'beige', '000000': 'black', '0000ff': 'blue',
    'a52a2a': 'brown', '008000': 'green', '808080': 'grey',
    '000000': 'multi', 'ffa500': 'orange', 'ffc0cb': 'pink',
    '800080': 'purple', 'ff0000': 'red', 'c0c0c0': 'silver',
    '40e0d0': 'turquoise', 'ffffff': 'white', 'ffff00': 'yellow'
}

@dataclass
class ProductResult:
    product_data: Dict
    color_distance: float
    dominant_color: tuple
    source: str

class ColorAnalyzer:
    def __init__(self):
        self.session = requests.Session()
        self.executor = ThreadPoolExecutor(max_workers=10)

    @staticmethod
    def hex_to_rgb(hex_code: str) -> tuple:
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def rgb_to_hex(rgb: tuple) -> str:
        return '%02x%02x%02x' % rgb

    @staticmethod
    def rgb_to_lab(rgb: tuple) -> tuple:
        rgb_normalized = [x/255 for x in rgb]
        
        def transform(c):
            if c > 0.04045:
                return ((c + 0.055) / 1.055) ** 2.4
            return c / 12.92
        
        rgb_transformed = [transform(c) for c in rgb_normalized]
        
        x = rgb_transformed[0] * 0.4124 + rgb_transformed[1] * 0.3576 + rgb_transformed[2] * 0.1805
        y = rgb_transformed[0] * 0.2126 + rgb_transformed[1] * 0.7152 + rgb_transformed[2] * 0.0722
        z = rgb_transformed[0] * 0.0193 + rgb_transformed[1] * 0.1192 + rgb_transformed[2] * 0.9505
        
        xn, yn, zn = 0.95047, 1.0, 1.08883
        
        def f(t):
            if t > 0.008856:
                return t ** (1/3)
            return 7.787 * t + 16/116
        
        fx = f(x/xn)
        fy = f(y/yn)
        fz = f(z/zn)
        
        L = max(0, 116 * fy - 16)
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        
        return (L, a, b)

    def get_center_cropped_image(self, image_url: str) -> Optional[Image.Image]:
        try:
            response = self.session.get(image_url, timeout=5)
            img = Image.open(BytesIO(response.content))
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            width, height = img.size
            crop_size = min(width, height) // 2
            
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            
            return img.crop((left, top, right, bottom))
        except Exception as e:
            print(f"Error processing image {image_url}: {str(e)}")
            return None

    @staticmethod
    def get_dominant_color(image: Image.Image) -> tuple:
        image = image.resize((150, 150))
        pixels = np.float32(image).reshape(-1, 3)
        
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

        _, counts = np.unique(labels, return_counts=True)
        
        valid_colors = []
        valid_counts = []
        
        for i, color in enumerate(palette):
            r, g, b = [int(c) for c in color]
            brightness = (r + g + b) / 3
            if 20 < brightness < 235:
                valid_colors.append(color)
                valid_counts.append(counts[i])
        
        if not valid_colors:
            return tuple(int(c) for c in palette[counts.argmax()])
        
        dominant = valid_colors[np.array(valid_counts).argmax()]
        return tuple(int(c) for c in dominant)

    @staticmethod
    def color_distance(color1: tuple, color2: tuple) -> float:
        lab1 = ColorAnalyzer.rgb_to_lab(color1)
        lab2 = ColorAnalyzer.rgb_to_lab(color2)
        
        delta_L = lab1[0] - lab2[0]
        delta_a = lab1[1] - lab2[1]
        delta_b = lab1[2] - lab2[2]
        
        return np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

    @staticmethod
    def find_closest_catalog_color(target_rgb: tuple, color_dict: dict) -> str:
        min_distance = float('inf')
        closest_color = None
        
        for hex_code in color_dict.keys():
            catalog_rgb = ColorAnalyzer.hex_to_rgb(hex_code)
            distance = ColorAnalyzer.color_distance(target_rgb, catalog_rgb)
            
            if distance < min_distance:
                min_distance = distance
                closest_color = hex_code
        
        return closest_color

class ProductFetcher:
    def __init__(self):
        self.color_analyzer = ColorAnalyzer()
        self.mongodb_handler = mongodb_handler

    def process_product(self, product: Dict, target_color: tuple, source: str) -> Optional[ProductResult]:
        try:
            image_url = product.get('searchImage')
            if not image_url:
                return None

            img = self.color_analyzer.get_center_cropped_image(image_url)
            
            if img:
                dominant_color = self.color_analyzer.get_dominant_color(img)
                distance = self.color_analyzer.color_distance(target_color, dominant_color)
                return ProductResult(product, distance, dominant_color, source)
            return None
        except Exception as e:
            print(f"Error processing product {product.get('productId', 'unknown')}: {str(e)}")
            return None

    def process_products_parallel(self, products: List[Dict], target_color: tuple, source: str) -> List[ProductResult]:
        process_func = partial(self.process_product, target_color=target_color, source=source)
        results = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_product = {executor.submit(process_func, product): product for product in products}
            for future in as_completed(future_to_product):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
        
        return results

    def get_myntra_products(self, keyword: str, hex_code: str) -> List[ProductResult]:
        try:
            target_rgb = self.color_analyzer.hex_to_rgb(hex_code)
            closest_hex = self.color_analyzer.find_closest_catalog_color(target_rgb, MYNTRA_COLORS)
            color = MYNTRA_COLORS[closest_hex]
            
            url = f'https://www.myntra.com/{keyword}?f=Color%3A{color}_0074D9&rawQuery={keyword}'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://www.myntra.com/',
                'Accept': '*/*'
            }
            
            response = requests.get(url, headers=headers)
            
            data = response.text
            data = data[data.find('window.__myx ='):]
            data = data[:data.find('</script>')]
            data = data[data.find('{'):data.rfind('}')+1]
            products = json.loads(data)['searchData']['results']['products']
            
            return self.process_products_parallel(products, target_rgb, 'myntra')
            
        except Exception as e:
            print(f"Error fetching Myntra products: {str(e)}")
            return []

    def get_hm_products(self, keyword: str, hex_code: str) -> List[ProductResult]:
        try:
            target_rgb = self.color_analyzer.hex_to_rgb(hex_code)
            closest_hex = self.color_analyzer.find_closest_catalog_color(target_rgb, HM_COLORS)
            color = HM_COLORS[closest_hex]
            
            base_url = 'https://www2.hm.com/en_in/search-results.html'
            params = {
                'q': keyword,
                'color': f"{color}_{closest_hex}" if color else None
            }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }

            response = requests.get(base_url, params=params, headers=headers)
            
            text = response.text
            data = text[text.find('"hits":'):text.find('"totalHits":')]
            data = data[data.find('['):data.rfind(']')+1]
            raw_products = json.loads(data)
            
            formatted_products = []
            image_prefix = 'https://image.hm.com/'
            
            for product in raw_products:
                try:
                    base_price = next((float(price['price']) for price in product.get('prices', [])
                                     if price['priceType'] == 'whitePrice'), None)
                    sale_price = next((float(price['price']) for price in product.get('prices', [])
                                     if price['priceType'] == 'yellowPrice'), None)
                    
                    if base_price and sale_price:
                        discount_percentage = round(((base_price - sale_price) / base_price) * 100)
                        discount = base_price - sale_price
                    else:
                        sale_price = base_price
                        discount = 0
                        discount_percentage = 0

                    formatted_product = {
                        'productId': product.get('articleCode', ''),
                        'product': product.get('title', ''),
                        'productName': product.get('title', ''),
                        'brand': 'H&M',
                        'searchImage': image_prefix + product.get('imageProductSrc', ''),
                        'mrp': base_price or sale_price or 0,
                        'price': sale_price or base_price or 0,
                        'discount': discount,
                        'discountDisplayLabel': f"({discount_percentage}% OFF)" if discount_percentage > 0 else '',
                        'primaryColour': product.get('swatches', [{}])[0].get('colorName', ''),
                        'category': product.get('category', ''),
                        'inventoryInfo': [{
                            'inventory': 0 if product.get('isOutOfStock') else 100,
                            'available': not product.get('isOutOfStock', False)
                        }]
                    }
                    formatted_products.append(formatted_product)
                
                except Exception as e:
                    print(f"Error processing H&M product: {str(e)}")
                    continue

            return self.process_products_parallel(formatted_products, target_rgb, 'hm')

        except Exception as e:
            print(f"Error fetching H&M products: {str(e)}")
            return []

async def fetch_products_parallel(product_fetcher: ProductFetcher, keyword: str, color: str):
    with ThreadPoolExecutor() as executor:
        myntra_future = executor.submit(product_fetcher.get_myntra_products, keyword, color)
        hm_future = executor.submit(product_fetcher.get_hm_products, keyword, color)
        
        myntra_results = myntra_future.result()
        hm_results = hm_future.result()
        
        return myntra_results, hm_results

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URL = os.getenv('MONGODB_URL')
mongodb_handler = MongoDBHandler(MONGODB_URL)
product_fetcher = ProductFetcher()

@app.get("/search")
async def search_products(
    keyword: str = Query(..., description="Keyword for product search"),
    color: str = Query(..., description="Hex code for color filter"),
    limit: int = Query(20, description="Maximum number of products to return")
):
    try:
        cached_results = mongodb_handler.get_cached_results(keyword, color)
        if cached_results:
            return {
                "message": "Retrieved from cache",
                "target_color": f"#{color.lstrip('#')}",
                "total_products": len(cached_results),
                "products": cached_results,
                "cache_hit": True
            }
        
        myntra_results, hm_results = await fetch_products_parallel(product_fetcher, keyword, color)
        
        all_results = myntra_results + hm_results
        sorted_results = sorted(all_results, key=lambda x: x.color_distance)
        
        combined_products = []
        target_rgb = ColorAnalyzer.hex_to_rgb(color)
        
        for result in sorted_results:
            product = result.product_data.copy()
            product.update({
                'source': result.source,
                'color_match_score': round(100 * (1 - min(result.color_distance / 100, 1)), 2),
                'dominant_color': '#' + ColorAnalyzer.rgb_to_hex(result.dominant_color),
                'color_distance': round(result.color_distance, 2)
            })
            combined_products.append(product)
        
        mongodb_handler.cache_search_results(keyword, color, combined_products)
        
        return {
            "message": "Successfully retrieved products",
            "target_color": f"#{color.lstrip('#')}",
            "total_products": len(combined_products),
            "products": combined_products,
            "cache_hit": False
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)