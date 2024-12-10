import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension


def sendMarkdownEmail(recipient_email, subject, markdown_content):
    sender_email = "302busey@gmail.com"
    app_password = "niam ssqi fbhr uzpz"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    html_content = markdown.markdown(
        markdown_content,
        extensions=['fenced_code', CodeHiliteExtension(linenums=False, css_class='highlight')]
    )

    html_email = f"""
    <html>
    <head>
        <style>
            .highlight pre {{
                background-color: #f5f5f5; /* Light gray background */
                color: #333333; /* Dark text color */
                padding: 10px;
                border-radius: 5px;
                font-family: 'Courier New', Courier, monospace;
                line-height: 1.5; /* Increase line height */
                overflow-x: auto; /* Enable horizontal scrolling */
            }}
            .highlight code {{
                background-color: transparent;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Create email message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = recipient_email

    # Attach plain text and HTML versions
    msg.attach(MIMEText(markdown_content, "plain"))
    msg.attach(MIMEText(html_email, "html"))

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, app_password)  # Log in with Gmail account
        server.sendmail(sender_email, recipient_email, msg.as_string())
