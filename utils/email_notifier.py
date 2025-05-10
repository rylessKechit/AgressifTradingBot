"""
Module de notification par email
Gère l'envoi de notifications par email
"""
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config.settings import NOTIFICATIONS

logger = logging.getLogger(__name__)

class EmailNotifier:
    """
    Classe pour l'envoi de notifications par email
    """
    
    def __init__(self, smtp_server=None, smtp_port=None, sender_email=None, 
                receiver_email=None, password=None, enabled=None):
        """
        Initialise le notificateur par email
        
        Args:
            smtp_server (str, optional): Serveur SMTP
            smtp_port (int, optional): Port SMTP
            sender_email (str, optional): Email de l'expéditeur
            receiver_email (str, optional): Email du destinataire
            password (str, optional): Mot de passe SMTP
            enabled (bool, optional): Si les notifications sont activées
        """
        self.smtp_server = smtp_server or NOTIFICATIONS.get('email', {}).get('smtp_server')
        self.smtp_port = smtp_port or NOTIFICATIONS.get('email', {}).get('smtp_port', 587)
        self.sender_email = sender_email or NOTIFICATIONS.get('email', {}).get('sender_email')
        self.receiver_email = receiver_email or NOTIFICATIONS.get('email', {}).get('receiver_email')
        self.password = password or NOTIFICATIONS.get('email', {}).get('password')
        self.enabled = enabled if enabled is not None else NOTIFICATIONS.get('email', {}).get('enabled', False)
        
        if self.enabled:
            # Vérifier que les paramètres nécessaires sont définis
            if not all([self.smtp_server, self.smtp_port, self.sender_email, self.receiver_email, self.password]):
                logger.warning("Paramètres de notification par email incomplets, notifications désactivées")
                self.enabled = False
            else:
                logger.info("Notificateur par email initialisé avec succès")
    
    def send_notification(self, subject, message):
        """
        Envoie une notification par email
        
        Args:
            subject (str): Sujet de l'email
            message (str): Corps de l'email
            
        Returns:
            bool: True si l'email a été envoyé avec succès, False sinon
        """
        if not self.enabled:
            logger.debug("Notifications par email désactivées")
            return False
            
        try:
            # Créer le message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = subject
            
            # Ajouter le corps du message
            msg.attach(MIMEText(message, 'plain'))
            
            # Connexion au serveur SMTP
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.password)
            
            # Envoi de l'email
            text = msg.as_string()
            server.sendmail(self.sender_email, self.receiver_email, text)
            
            # Fermer la connexion
            server.quit()
            
            logger.info(f"Email envoyé avec succès à {self.receiver_email}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'email: {e}")
            return False
    
    def send_alert(self, title, message, level="info"):
        """
        Envoie une alerte par email
        
        Args:
            title (str): Titre de l'alerte
            message (str): Message de l'alerte
            level (str, optional): Niveau d'alerte ("info", "warning", "error", "critical")
            
        Returns:
            bool: True si l'alerte a été envoyée avec succès, False sinon
        """
        subject = f"[ALERTE {level.upper()}] {title}"
        full_message = f"Niveau: {level.upper()}\n\n{message}"
        
        return self.send_notification(subject, full_message)
    
    def send_trade_notification(self, action, symbol, side, price, amount, pnl=None):
        """
        Envoie une notification de trade
        
        Args:
            action (str): Action ("open", "close")
            symbol (str): Symbole de la paire
            side (str): Côté ("long", "short")
            price (float): Prix
            amount (float): Quantité
            pnl (float, optional): Profit/Perte
            
        Returns:
            bool: True si la notification a été envoyée avec succès, False sinon
        """
        if action == "open":
            subject = f"Nouvelle position: {side} {symbol}"
            message = f"Une nouvelle position a été ouverte:\n\n" \
                     f"Symbole: {symbol}\n" \
                     f"Direction: {side}\n" \
                     f"Prix: {price}\n" \
                     f"Quantité: {amount}\n" \
                     f"Valeur: {price * amount}"
        else:  # close
            subject = f"Position fermée: {side} {symbol}"
            message = f"Une position a été fermée:\n\n" \
                     f"Symbole: {symbol}\n" \
                     f"Direction: {side}\n" \
                     f"Prix: {price}\n" \
                     f"Quantité: {amount}\n"
            
            if pnl is not None:
                message += f"P/L: {pnl}\n"
                if pnl > 0:
                    message += "Résultat: GAIN"
                else:
                    message += "Résultat: PERTE"
        
        return self.send_notification(subject, message)
    
    def send_daily_report(self, stats):
        """
        Envoie un rapport quotidien
        
        Args:
            stats (dict): Statistiques du trading
            
        Returns:
            bool: True si le rapport a été envoyé avec succès, False sinon
        """
        subject = "Rapport quotidien de trading"
        
        message = "Voici le résumé de vos activités de trading:\n\n"
        
        # Capital et P/L
        message += f"Capital actuel: {stats.get('current_capital', 0)}\n"
        message += f"P/L aujourd'hui: {stats.get('daily_pnl', 0)}\n"
        message += f"P/L total: {stats.get('total_profit', 0)}\n\n"
        
        # Statistiques des trades
        message += f"Trades aujourd'hui: {stats.get('daily_trades', 0)}\n"
        message += f"Trades gagnants: {stats.get('daily_winning_trades', 0)}\n"
        message += f"Trades perdants: {stats.get('daily_losing_trades', 0)}\n"
        message += f"Taux de réussite: {stats.get('daily_win_rate', 0)}%\n\n"
        
        # Positions ouvertes
        open_positions = stats.get('open_positions', [])
        if open_positions:
            message += "Positions ouvertes:\n"
            for pos in open_positions:
                message += f"- {pos['symbol']} ({pos['side']}): entrée à {pos['entry_price']}, P/L actuel: {pos['unrealized_pnl']}\n"
        else:
            message += "Aucune position ouverte.\n"
        
        # Performances globales
        message += "\nPerformances globales:\n"
        message += f"Trades totaux: {stats.get('total_trades', 0)}\n"
        message += f"Taux de réussite global: {stats.get('win_rate', 0)}%\n"
        message += f"Ratio profit/perte: {stats.get('profit_factor', 0)}\n"
        message += f"Drawdown maximum: {stats.get('max_drawdown', 0)}%\n"
        
        return self.send_notification(subject, message)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialiser le notificateur
    notifier = EmailNotifier(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        sender_email="votre_email@gmail.com",
        receiver_email="destinataire@example.com",
        password="votre_mot_de_passe",
        enabled=True
    )
    
    # Envoyer une notification
    notifier.send_notification(
        "Test de notification",
        "Ceci est un test de notification par email."
    )
    
    # Envoyer une alerte
    notifier.send_alert(
        "Test d'alerte",
        "Ceci est un test d'alerte par email.",
        level="warning"
    )
    
    # Envoyer une notification de trade
    notifier.send_trade_notification(
        "open",
        "BTC/USDT",
        "long",
        50000,
        0.1
    )
    
    # Envoyer un rapport quotidien
    stats = {
        'current_capital': 10500,
        'daily_pnl': 500,
        'total_profit': 1500,
        'daily_trades': 5,
        'daily_winning_trades': 3,
        'daily_losing_trades': 2,
        'daily_win_rate': 60,
        'open_positions': [
            {'symbol': 'BTC/USDT', 'side': 'long', 'entry_price': 50000, 'unrealized_pnl': 200},
            {'symbol': 'ETH/USDT', 'side': 'short', 'entry_price': 3000, 'unrealized_pnl': -50}
        ],
        'total_trades': 50,
        'win_rate': 62,
        'profit_factor': 1.8,
        'max_drawdown': 12
    }
    
    notifier.send_daily_report(stats)