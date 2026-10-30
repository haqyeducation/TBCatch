"""A local-only, buildless research MVP using the existing Python environment."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse,base64,json,mimetypes,secrets,socket,ssl,threading
from urllib.parse import urlsplit
from inference import Predictor,PACKAGE

HERE=Path(__file__).resolve().parent
TOKEN=secrets.token_urlsafe(32)
predictor=Predictor()
MAX_BODY=8_500_000
ALLOWED_HOSTS={'localhost','127.0.0.1','::1',socket.gethostname().lower()}
try:ALLOWED_HOSTS.update(socket.gethostbyname_ex(socket.gethostname())[2])
except OSError:pass
FILES={'/':HERE/'static/index.html','/app.js':HERE/'static/app.js','/style.css':HERE/'static/style.css',
       '/check.css':HERE/'static/check.css','/motion.js':HERE/'static/motion.js',
       '/vendor/ppg.js':HERE/'static/vendor/ppg.js',
       '/vendor/LICENSE.txt':HERE/'static/vendor/LICENSE.txt'}
# Enumerate public assets only; model, config, and patient files stay outside this map.
FILES.update({'/assets/'+f.name:f for f in (HERE/'static/assets').iterdir() if f.is_file()})

class Handler(BaseHTTPRequestHandler):
    server_version='TBResearchMVP'
    def setup(self):super().setup();self.connection.settimeout(240)
    def log_message(self,fmt,*args):
        # Do not log audio, clinical inputs, or scores.
        if self.path!='/api/health':super().log_message(fmt,*args)
    def send(self,status,data,kind='application/json; charset=utf-8'):
        payload=json.dumps(data,allow_nan=False).encode() if isinstance(data,(dict,list)) else data
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(payload)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer');self.send_header('Permissions-Policy','camera=(self), microphone=(self)')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:self.wfile.write(payload)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        if urlsplit('//'+self.headers.get('Host','')).hostname not in ALLOWED_HOSTS:return self.send(421,{'error':'Server address is not allowed.'})
        path=urlsplit(self.path).path
        if path=='/api/health':return self.send(200,predictor.status())
        if path=='/api/demo':return self.send(200,json.loads((HERE/'demo.json').read_text()))
        if path in FILES:
            f=FILES[path]
            if not f.is_file():return self.send(404,{'error':'App asset not found.'})
            data=f.read_bytes()
            if path=='/':data=data.replace(b'__API_TOKEN__',TOKEN.encode())
            return self.send(200,data,mimetypes.guess_type(str(f))[0] or 'application/octet-stream')
        return self.send(404,{'error':'Page not found.'})
    def do_POST(self):
        if urlsplit('//'+self.headers.get('Host','')).hostname not in ALLOWED_HOSTS:return self.send(421,{'error':'Server address is not allowed.'})
        if self.path!='/api/analyze':return self.send(404,{'error':'Endpoint not found.'})
        origin=self.headers.get('Origin');scheme='https' if isinstance(self.connection,ssl.SSLSocket) else 'http'
        if self.headers.get('X-MVP-Key')!=TOKEN or origin!=f'{scheme}://{self.headers.get("Host")}':return self.send(403,{'error':'Request rejected. Reopen the app page.'})
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.send(415,{'error':'Request format must be JSON.'})
        try:length=int(self.headers.get('Content-Length','0'))
        except ValueError:return self.send(400,{'error':'Invalid request size.'})
        if not 0<length<=MAX_BODY:return self.send(413,{'error':'Audio is too large. Use a recording up to 30 seconds.'})
        if not predictor.lock.acquire(blocking=False):return self.send(409,{'error':'Another check-in is being processed. Try again in a moment.'})
        try:
            request=json.loads(self.rfile.read(length))
            if not isinstance(request,dict):raise ValueError('Invalid request.')
            if not predictor.ready:return self.send(503,{'error':predictor.error or 'The model is loading. Wait for the ready status.'})
            return self.send(200,predictor.analyze(request))
        except (ValueError,TypeError,KeyError) as exc:return self.send(400,{'error':str(exc)})
        except Exception:
            import traceback
            traceback.print_exc();return self.send(500,{'error':'Analysis did not finish. Try again or check the server terminal.'})
        finally:predictor.lock.release()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--cert');parser.add_argument('--key');parser.add_argument('--http-port',type=int);args=parser.parse_args()
    if bool(args.cert)!=bool(args.key):parser.error('Sediakan --cert dan --key bersamaan.')
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    if args.cert:
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);context.minimum_version=ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.cert,args.key);server.socket=context.wrap_socket(server.socket,server_side=True)
    if args.http_port:
        if not args.cert:parser.error('--http-port digunakan bersama server HTTPS.')
        local=ThreadingHTTPServer(('127.0.0.1',args.http_port),Handler)
        threading.Thread(target=local.serve_forever,daemon=True).start()
        print(f'Untuk laptop: http://localhost:{args.http_port}',flush=True)
    threading.Thread(target=predictor.initialize,daemon=True).start()
    print(f'Buka {"https" if args.cert else "http"}://localhost:{args.port}',flush=True)
    print('Model dimuat di latar belakang. Status terlihat di aplikasi. Ctrl+C untuk berhenti.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close()
        if args.http_port:local.shutdown();local.server_close()

if __name__=='__main__':main()
