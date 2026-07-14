import numpy as np, pandas as pd, os, json, sys, urllib.request, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

BASE = os.path.dirname(os.path.abspath(__file__))
PORT = 8765

FEATURES = ["close","high","low","volume","ma10","ma20","bb_upper","bb_lower","ema10","obv"]

print("Lade Daten...")
df = pd.read_csv(os.path.join(BASE, "btcusd_binance_enriched.csv"))
df = df[df['time'] >= 1725667200].reset_index(drop=True)

raw = df[FEATURES].values.astype(np.float32)
data = np.zeros_like(raw)
data[1:,0] = np.log(np.maximum(raw[1:,0],1e-10)/np.maximum(raw[:-1,0],1e-10))
pc = raw[:,0]
data[1:,1] = np.log(np.maximum(raw[1:,1],1e-10)/np.maximum(pc[1:],1e-10))
data[1:,2] = np.log(np.maximum(pc[1:],1e-10)/np.maximum(raw[1:,2],1e-10))
for ci in [3,4,5,6,7,8]:
    data[1:,ci] = np.log(np.maximum(raw[1:,ci],1e-10)/np.maximum(raw[:-1,ci],1e-10))
data[1:,9] = raw[1:,9]

n_total = len(df)
n_train = int(n_total * 0.7)
n_val = int(n_total * 0.15)
test_start = n_train + n_val

print("Lade Modelle...")
def get_latest(folder, prefix):
    d = os.path.join(BASE, folder)
    eps = sorted([f for f in os.listdir(d) if f.startswith(prefix) and f.endswith('.keras') and '_ep' in f],
                 key=lambda x: int(x.split('_ep')[1].split('.')[0]))
    if eps: return os.path.join(d, eps[-1])
    best = os.path.join(d, f"{prefix}.keras")
    return best if os.path.exists(best) else None

models = {}
for name, folder, pref in [('close','models','tradenetV3_btc'), ('vol','models','tradenetV3_volume'), ('hl','models','tradenetV3_hl')]:
    p = get_latest(folder, pref)
    if p:
        models[name] = tf.keras.models.load_model(p, compile=False)
        print(f"  {name}: {os.path.relpath(p, BASE)}")
    else:
        print(f"  FEHLER: {name} nicht gefunden"); sys.exit(1)

W = models['close'].input_shape[1]
print(f"Modelle geladen. Window={W}, Features={models['close'].input_shape[2]}")

def compute_features(ohlcv):
    N = len(ohlcv); f = np.zeros((N,10), dtype=np.float64)
    f[:,:4] = ohlcv[:,:4]
    for i in range(N):
        if i >= 9: f[i,4] = np.mean(f[i-9:i+1,0])
        else: f[i,4] = np.mean(f[:i+1,0])
        s = max(i,19); ma20 = np.mean(f[i-s:i+1,0]) if i>=19 else np.mean(f[:i+1,0])
        std20 = np.std(f[i-s:i+1,0]) if i>=19 else (np.std(f[:i+1,0]) if i>0 else 0)
        f[i,5] = ma20; f[i,6] = ma20+2*std20; f[i,7] = ma20-2*std20
    f[0,8] = f[0,0]
    for i in range(1,N): f[i,8] = 0.9*f[i-1,8] + 0.1*f[i,0]
    f[0,9] = 0
    for i in range(1,N):
        if f[i,0] > f[i-1,0]: f[i,9] = f[i-1,9] + f[i,3]
        elif f[i,0] < f[i-1,0]: f[i,9] = f[i-1,9] - f[i,3]
        else: f[i,9] = f[i-1,9]
    return f.astype(np.float32)

def norm_features(f):
    N = len(f); d = np.zeros_like(f); pc = f[:,0]
    d[1:,0] = np.log(np.maximum(f[1:,0],1e-10)/np.maximum(f[:-1,0],1e-10))
    d[1:,1] = np.log(np.maximum(f[1:,1],1e-10)/np.maximum(pc[1:],1e-10))
    d[1:,2] = np.log(np.maximum(pc[1:],1e-10)/np.maximum(f[1:,2],1e-10))
    for ci in range(3,9): d[1:,ci] = np.log(np.maximum(f[1:,ci],1e-10)/np.maximum(f[:-1,ci],1e-10))
    d[:,9] = f[:,9]
    return d.astype(np.float32)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global html
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/' or path == '/trading_ui.html':
            with open(os.path.join(BASE, 'trading_ui.html'), 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)

        elif path == '/api/export_csv':
            idx = int(params.get('idx', [test_start + W + 50])[0])
            n_pred = int(params.get('n', [100])[0])
            try:
                raw_roll = raw[idx - W + 1: idx + 1].copy()
                data_roll = data[idx - W + 1: idx + 1].copy()
                lines = ["Date,Open,High,Low,Close,Volume"]
                base_time = int(df.iloc[idx]['time'])
                for step in range(n_pred):
                    win = data_roll[-W:]
                    wm = np.mean(win, axis=0, keepdims=True)
                    ws = np.maximum(np.std(win, axis=0, keepdims=True), 1e-6)
                    x = ((win - wm) / ws).reshape(1, W, -1)
                    p_c = models['close'](x, training=False).numpy()[0,0]
                    p_h = models['hl'](x, training=False).numpy()[0]
                    p_v = models['vol'](x, training=False).numpy()[0,0]
                    pred_close = raw_roll[-1,0] * np.exp(p_c * ws[0,0] + wm[0,0])
                    pred_high = pred_close * np.exp(p_h[0] * ws[0,1] + wm[0,1])
                    pred_low = pred_close / np.exp(p_h[1] * ws[0,2] + wm[0,2])
                    pred_vol = raw_roll[-1,3] * np.exp(p_v * ws[0,3] + wm[0,3])
                    t = base_time + (step + 1) * 60
                    ts = pd.Timestamp(t, unit='s').strftime('%Y-%m-%d %H:%M:%S')
                    lines.append(f"{ts},{pred_close:.1f},{pred_high:.1f},{pred_low:.1f},{pred_close:.1f},{pred_vol:.1f}")
                    new_raw = raw_roll[-1].copy()
                    new_raw[0:4] = [pred_close, pred_high, pred_low, pred_vol]
                    all_c = list(raw_roll[-19:,0]) + [pred_close]
                    new_raw[4] = np.mean(all_c[-10:]); new_raw[5] = np.mean(all_c[-20:])
                    s = np.std(all_c[-20:]) if len(all_c)>=20 else np.std(all_c)
                    new_raw[6] = new_raw[5] + 2*s; new_raw[7] = new_raw[5] - 2*s
                    new_raw[8] = 0.9*raw_roll[-1,8] + 0.1*pred_close
                    new_raw[9] = raw_roll[-1,9] + pred_vol*(1 if pred_close>=raw_roll[-1,0] else -1)
                    raw_roll = np.vstack([raw_roll, new_raw.reshape(1,-1)])
                    nl = np.zeros((1,len(FEATURES)),dtype=np.float32)
                    nl[0,0] = np.log(max(new_raw[0],1e-10)/max(raw_roll[-2,0],1e-10))
                    nl[0,1] = np.log(max(new_raw[1],1e-10)/max(new_raw[0],1e-10))
                    nl[0,2] = np.log(max(new_raw[0],1e-10)/max(new_raw[2],1e-10))
                    for ci in [3,4,5,6,7,8]:
                        nl[0,ci] = np.log(max(new_raw[ci],1e-10)/max(raw_roll[-2,ci],1e-10))
                    nl[0,9] = new_raw[9]
                    data_roll = np.vstack([data_roll, nl])
                content = "\n".join(lines).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', f'attachment; filename="ai_pred_{idx}.csv"')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/data':
            offset = int(params.get('offset', [0])[0])
            limit = min(int(params.get('limit', [200])[0]), 1000)
            end = min(offset + limit, n_total)
            rows = []
            for i in range(offset, end):
                rows.append({
                    'time': int(df.iloc[i]['time']),
                    'open': float(df.iloc[i]['open']),
                    'high': float(df.iloc[i]['high']),
                    'low': float(df.iloc[i]['low']),
                    'close': float(df.iloc[i]['close']),
                    'volume': float(df.iloc[i]['volume']),
                })
            self.send_json(rows)

        elif path == '/api/predict':
            idx = int(params.get('idx', [test_start + W + 50])[0])
            n_pred = int(params.get('n', [1])[0])
            try:
                raw_roll = raw[idx - W + 1: idx + 1].copy()
                data_roll = data[idx - W + 1: idx + 1].copy()
                preds = []
                for step in range(n_pred):
                    win = data_roll[-W:]
                    wm = np.mean(win, axis=0, keepdims=True)
                    ws = np.maximum(np.std(win, axis=0, keepdims=True), 1e-6)
                    x = ((win - wm) / ws).reshape(1, W, -1)
                    p_c = models['close'](x, training=False).numpy()[0,0]
                    p_h = models['hl'](x, training=False).numpy()[0]
                    p_v = models['vol'](x, training=False).numpy()[0,0]
                    pred_close = raw_roll[-1,0] * np.exp(p_c * ws[0,0] + wm[0,0])
                    pred_high = pred_close * np.exp(p_h[0] * ws[0,1] + wm[0,1])
                    pred_low = pred_close / np.exp(p_h[1] * ws[0,2] + wm[0,2])
                    pred_vol = raw_roll[-1,3] * np.exp(p_v * ws[0,3] + wm[0,3])
                    preds.append({'close':float(pred_close),'high':float(pred_high),'low':float(pred_low),'volume':float(pred_vol)})
                    new_raw = raw_roll[-1].copy()
                    new_raw[0:4] = [pred_close, pred_high, pred_low, pred_vol]
                    all_c = list(raw_roll[-19:,0]) + [pred_close]
                    new_raw[4] = np.mean(all_c[-10:])
                    new_raw[5] = np.mean(all_c[-20:])
                    s = np.std(all_c[-20:]) if len(all_c)>=20 else np.std(all_c)
                    new_raw[6] = new_raw[5] + 2*s
                    new_raw[7] = new_raw[5] - 2*s
                    new_raw[8] = 0.9*raw_roll[-1,8] + 0.1*pred_close
                    new_raw[9] = raw_roll[-1,9] + pred_vol*(1 if pred_close>=raw_roll[-1,0] else -1)
                    raw_roll = np.vstack([raw_roll, new_raw.reshape(1,-1)])
                    nl = np.zeros((1,len(FEATURES)),dtype=np.float32)
                    nl[0,0] = np.log(max(new_raw[0],1e-10)/max(raw_roll[-2,0],1e-10))
                    nl[0,1] = np.log(max(new_raw[1],1e-10)/max(new_raw[0],1e-10))
                    nl[0,2] = np.log(max(new_raw[0],1e-10)/max(new_raw[2],1e-10))
                    for ci in [3,4,5,6,7,8]:
                        nl[0,ci] = np.log(max(new_raw[ci],1e-10)/max(raw_roll[-2,ci],1e-10))
                    nl[0,9] = new_raw[9]
                    data_roll = np.vstack([data_roll, nl])
                self.send_json(preds)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/tv_data':
            try:
                idx = n_total - W - 100
                raw_roll = raw[idx - W + 1: idx + 1].copy()
                data_roll = data[idx - W + 1: idx + 1].copy()
                times = [int(df.iloc[i]['time']) for i in range(idx - W + 1, idx + 1)]
                opens_o = [float(df.iloc[i]['open']) for i in range(idx - W + 1, idx + 1)]
                highs_o = [float(df.iloc[i]['high']) for i in range(idx - W + 1, idx + 1)]
                lows_o = [float(df.iloc[i]['low']) for i in range(idx - W + 1, idx + 1)]
                closes_o = [float(df.iloc[i]['close']) for i in range(idx - W + 1, idx + 1)]
                vols_o = [float(df.iloc[i]['volume']) for i in range(idx - W + 1, idx + 1)]
                for step in range(100):
                    win = data_roll[-W:]
                    wm = np.mean(win, axis=0, keepdims=True)
                    ws = np.maximum(np.std(win, axis=0, keepdims=True), 1e-6)
                    x = ((win - wm) / ws).reshape(1, W, -1)
                    p_c = models['close'](x, training=False).numpy()[0,0]
                    p_h = models['hl'](x, training=False).numpy()[0]
                    pred_close = raw_roll[-1,0] * np.exp(p_c * ws[0,0] + wm[0,0])
                    pred_high = pred_close * np.exp(p_h[0] * ws[0,1] + wm[0,1])
                    pred_low = pred_close / np.exp(p_h[1] * ws[0,2] + wm[0,2])
                    t = int(df.iloc[idx]['time']) + (step + 1) * 60
                    times.append(t)
                    opens_o.append(float(pred_close))
                    highs_o.append(float(pred_high))
                    lows_o.append(float(pred_low))
                    closes_o.append(float(pred_close))
                    vols_o.append(1000.0)
                    new_raw = raw_roll[-1].copy()
                    new_raw[0:4] = [pred_close, pred_high, pred_low, pred_close]
                    all_c = list(raw_roll[-19:,0]) + [pred_close]
                    new_raw[4] = np.mean(all_c[-10:]); new_raw[5] = np.mean(all_c[-20:])
                    s = np.std(all_c[-20:]) if len(all_c)>=20 else np.std(all_c)
                    new_raw[6] = new_raw[5] + 2*s; new_raw[7] = new_raw[5] - 2*s
                    new_raw[8] = 0.9*raw_roll[-1,8] + 0.1*pred_close
                    new_raw[9] = raw_roll[-1,9] + pred_close*(1 if pred_close>=raw_roll[-1,0] else -1)
                    raw_roll = np.vstack([raw_roll, new_raw.reshape(1,-1)])
                    nl = np.zeros((1,len(FEATURES)),dtype=np.float32)
                    nl[0,0] = np.log(max(new_raw[0],1e-10)/max(raw_roll[-2,0],1e-10))
                    nl[0,1] = np.log(max(new_raw[1],1e-10)/max(new_raw[0],1e-10))
                    nl[0,2] = np.log(max(new_raw[0],1e-10)/max(new_raw[2],1e-10))
                    for ci in [3,4,5,6,7,8]:
                        nl[0,ci] = np.log(max(new_raw[ci],1e-10)/max(raw_roll[-2,ci],1e-10))
                    nl[0,9] = new_raw[9]
                    data_roll = np.vstack([data_roll, nl])
                tv_data = {"s":"ok","t":times,"o":opens_o,"h":highs_o,"l":lows_o,"c":closes_o,"v":vols_o}
                self.send_json(tv_data)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/predict_live':
            try:
                idx = n_total - W - 10
                raw_roll = raw[idx - W + 1: idx + 1].copy()
                data_roll = data[idx - W + 1: idx + 1].copy()
                base_time = int(df.iloc[idx]['time'])
                preds_list = []
                for step in range(100):
                    win = data_roll[-W:]
                    wm = np.mean(win, axis=0, keepdims=True)
                    ws = np.maximum(np.std(win, axis=0, keepdims=True), 1e-6)
                    x = ((win - wm) / ws).reshape(1, W, -1)
                    p_c = models['close'](x, training=False).numpy()[0,0]
                    p_h = models['hl'](x, training=False).numpy()[0]
                    pred_close = float(raw_roll[-1,0] * np.exp(p_c * ws[0,0] + wm[0,0]))
                    pred_high = float(pred_close * np.exp(p_h[0] * ws[0,1] + wm[0,1]))
                    pred_low = float(pred_close / np.exp(p_h[1] * ws[0,2] + wm[0,2]))
                    t = base_time + (step + 1) * 60
                    preds_list.append({"t": t, "o": pred_close, "h": pred_high, "l": pred_low, "c": pred_close})
                    new_raw = raw_roll[-1].copy()
                    new_raw[0:4] = [pred_close, pred_high, pred_low, pred_close]
                    all_c = list(raw_roll[-19:,0]) + [pred_close]
                    new_raw[4] = np.mean(all_c[-10:]); new_raw[5] = np.mean(all_c[-20:])
                    s = np.std(all_c[-20:]) if len(all_c)>=20 else np.std(all_c)
                    new_raw[6] = new_raw[5] + 2*s; new_raw[7] = new_raw[5] - 2*s
                    new_raw[8] = 0.9*raw_roll[-1,8] + 0.1*pred_close
                    new_raw[9] = raw_roll[-1,9] + pred_close*(1 if pred_close>=raw_roll[-1,0] else -1)
                    raw_roll = np.vstack([raw_roll, new_raw.reshape(1,-1)])
                    nl = np.zeros((1,len(FEATURES)),dtype=np.float32)
                    nl[0,0] = np.log(max(new_raw[0],1e-10)/max(raw_roll[-2,0],1e-10))
                    nl[0,1] = np.log(max(new_raw[1],1e-10)/max(new_raw[0],1e-10))
                    nl[0,2] = np.log(max(new_raw[0],1e-10)/max(new_raw[2],1e-10))
                    for ci in [3,4,5,6,7,8]:
                        nl[0,ci] = np.log(max(new_raw[ci],1e-10)/max(raw_roll[-2,ci],1e-10))
                    nl[0,9] = new_raw[9]
                    data_roll = np.vstack([data_roll, nl])
                self.send_json({"time": base_time, "close": float(df.iloc[idx]['close']), "preds": preds_list})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/binance_klines':
            limit = min(int(params.get('limit', [100])[0]), 500)
            try:
                url = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}'
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    klines = json.loads(resp.read())
                rows = []
                for k in klines:
                    rows.append({'time': int(k[0])//1000, 'open': float(k[1]), 'high': float(k[2]),
                                 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])})
                self.send_json(rows)
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/api/binance_predict':
            try:
                limit = 300
                url = f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit={limit}'
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    klines = json.loads(resp.read())
                ohlcv = np.array([[float(k[4]), float(k[2]), float(k[3]), float(k[5])] for k in klines], dtype=np.float64)
                base_time = int(klines[-1][0])//1000
                feat = compute_features(ohlcv)
                nd = norm_features(feat)
                raw_roll = feat[-W:].copy()
                data_roll = nd[-W:].copy()
                preds_list = []
                for step in range(100):
                    win = data_roll[-W:]
                    wm = np.mean(win, axis=0, keepdims=True)
                    ws = np.maximum(np.std(win, axis=0, keepdims=True), 1e-6)
                    x = ((win - wm) / ws).reshape(1, W, -1)
                    p_c = models['close'](x, training=False).numpy()[0,0]
                    p_h = models['hl'](x, training=False).numpy()[0]
                    pred_close = float(raw_roll[-1,0] * np.exp(p_c * ws[0,0] + wm[0,0]))
                    pred_high = float(pred_close * np.exp(p_h[0] * ws[0,1] + wm[0,1]))
                    pred_low = float(pred_close / np.exp(p_h[1] * ws[0,2] + wm[0,2]))
                    t = base_time + (step + 1) * 60
                    preds_list.append({"t": t, "o": pred_close, "h": pred_high, "l": pred_low, "c": pred_close})
                    new_raw = raw_roll[-1].copy()
                    new_raw[0:4] = [pred_close, pred_high, pred_low, pred_close]
                    all_c = list(raw_roll[-19:,0]) + [pred_close]
                    new_raw[4] = np.mean(all_c[-10:]); new_raw[5] = np.mean(all_c[-20:])
                    s = np.std(all_c[-20:]) if len(all_c)>=20 else np.std(all_c)
                    new_raw[6] = new_raw[5] + 2*s; new_raw[7] = new_raw[5] - 2*s
                    new_raw[8] = 0.9*raw_roll[-1,8] + 0.1*pred_close
                    new_raw[9] = raw_roll[-1,9] + pred_close*(1 if pred_close>=raw_roll[-1,0] else -1)
                    raw_roll = np.vstack([raw_roll, new_raw.reshape(1,-1)])
                    nl = np.zeros((1,len(FEATURES)),dtype=np.float32)
                    nl[0,0] = np.log(max(new_raw[0],1e-10)/max(raw_roll[-2,0],1e-10))
                    nl[0,1] = np.log(max(new_raw[1],1e-10)/max(new_raw[0],1e-10))
                    nl[0,2] = np.log(max(new_raw[0],1e-10)/max(new_raw[2],1e-10))
                    for ci in [3,4,5,6,7,8]:
                        nl[0,ci] = np.log(max(new_raw[ci],1e-10)/max(raw_roll[-2,ci],1e-10))
                    nl[0,9] = new_raw[9]
                    data_roll = np.vstack([data_roll, nl])
                self.send_json({"time": base_time, "close": float(klines[-1][4]), "preds": preds_list})
            except Exception as e:
                self.send_json({'error': str(e)})

        elif path == '/live_chart':
            with open(os.path.join(BASE, 'live_chart.html'), 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(content)

        elif path == '/api/info':
            self.send_json({
                'total_rows': n_total,
                'test_start': test_start,
                'window': W,
                'features': FEATURES,
                'models': list(models.keys()),
            })

        else:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass

print(f"\nServer startet auf http://localhost:{PORT}")
print("Browser öffnen...\n")
import webbrowser
webbrowser.open(f'http://localhost:{PORT}/trading_ui.html')
HTTPServer(('', PORT), Handler).serve_forever()
