import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_HOST = "srv1599835.hstgr.cloud"
SMTP_PORT = 3025

msg = MIMEMultipart()
msg['From'] = "seguranca@banco-falso.tk"
msg['To'] = "inbox@teste.local"
msg['Subject'] = "⚠️ Ação urgente na sua conta"
msg.attach(MIMEText(
    "Clique aqui para evitar o bloqueio: http://site-phishing-falso.com/login?token=abc123",
    'plain'
))

with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
    server.sendmail(msg['From'], [msg['To']], msg.as_string())
    print("Email enviado com sucesso!")