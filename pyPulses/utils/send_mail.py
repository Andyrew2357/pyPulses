from email.message import EmailMessage
import configparser
import smtplib
from typing import List

def sendMail(subject: str, message: str, email_cfg: str,
             recipients: str | List[str] = 'all'):
    """
    Send an email (with certain carriers, you can also use 'text by mail').

    Parameters
    ----------
    subject : str
    message : str
    email_cfg : str
        path to an .ini containing contact info
    recipients : str or list of str, default='all'
        recipients to send the email; by default it is sent to everyone in the 
        .ini.
    """

    cfg = configparser.ConfigParser()
    cfg.read(email_cfg)

    if recipients == 'all':
        recipients = [cfg['Recipients'][i] for i in cfg['Recipients']]
    else:
        if type(recipients) == str:
            recipients = (recipients,)
        recipients = [cfg['Recipients'][i] for i in recipients]

    host = cfg['Sender']['host']
    port = int(cfg['Sender']['port'])
    address = cfg['Sender']['address']
    password = cfg['Sender']['pwd']

    server = smtplib.SMTP(host, port)
    server.starttls()
    server.login(address, password) 

    for rec in recipients:
        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = subject
        msg['From'] = address
        msg['To'] = rec
        server.send_message(msg)
