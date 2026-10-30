"""Create an untrusted local HTTPS certificate; never install a root CA."""
from pathlib import Path
import datetime,ipaddress,socket
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import rsa

out=Path(__file__).resolve().parent/'.local';out.mkdir(exist_ok=True)
if (out/'cert.pem').exists() or (out/'key.pem').exists():
    if not (out/'cert.pem').is_file() or not (out/'key.pem').is_file():raise RuntimeError('Sertifikat tidak lengkap. Periksa folder .local.')
    print('Menggunakan sertifikat lokal yang sudah ada.')
else:
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    ips={'127.0.0.1'}
    try:ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:pass
    subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'TBCatch Local Research MVP')])
    now=datetime.datetime.now(datetime.timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(subject).issuer_name(subject).public_key(key.public_key())
          .serial_number(x509.random_serial_number()).not_valid_before(now-datetime.timedelta(minutes=5))
          .not_valid_after(now+datetime.timedelta(days=90)).add_extension(x509.SubjectAlternativeName(
              [x509.DNSName('localhost'),x509.DNSName(socket.gethostname())]+[x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]),critical=False)
          .sign(key,hashes.SHA256()))
    (out/'key.pem').write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    (out/'cert.pem').write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    print('Sertifikat self-signed dibuat untuk localhost dan IP LAN. Root CA tidak dipasang.')
for ip in socket.gethostbyname_ex(socket.gethostname())[2]:print(f'Untuk HP satu jaringan: https://{ip}:8443')
