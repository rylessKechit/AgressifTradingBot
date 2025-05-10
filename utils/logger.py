"""
Module de configuration du logging
Gère la configuration et l'initialisation du système de journalisation
"""
import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

def setup_logger(level=logging.INFO, format_str=None, log_file=None, max_bytes=10485760, backup_count=5):
    """
    Configure et initialise le système de logging
    
    Args:
        level (int, optional): Niveau de journalisation
        format_str (str, optional): Format des messages
        log_file (str, optional): Chemin du fichier de log
        max_bytes (int, optional): Taille maximale du fichier de log
        backup_count (int, optional): Nombre de fichiers de sauvegarde
        
    Returns:
        logging.Logger: Logger configuré
    """
    # Convertir le niveau de log si c'est une chaîne
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Format par défaut
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Créer le formatter
    formatter = logging.Formatter(format_str)
    
    # Créer le logger root
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Supprimer les handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Ajouter un handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Ajouter un handler pour le fichier si spécifié
    if log_file:
        # Créer le répertoire si nécessaire
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Utiliser un RotatingFileHandler pour limiter la taille des fichiers
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def setup_logger_with_rotation(level=logging.INFO, format_str=None, log_file=None, when='midnight', interval=1, backup_count=7):
    """
    Configure et initialise le système de logging avec rotation temporelle
    
    Args:
        level (int, optional): Niveau de journalisation
        format_str (str, optional): Format des messages
        log_file (str, optional): Chemin du fichier de log
        when (str, optional): Quand faire la rotation ('S', 'M', 'H', 'D', 'midnight')
        interval (int, optional): Intervalle de rotation
        backup_count (int, optional): Nombre de fichiers de sauvegarde
        
    Returns:
        logging.Logger: Logger configuré
    """
    # Convertir le niveau de log si c'est une chaîne
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Format par défaut
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Créer le formatter
    formatter = logging.Formatter(format_str)
    
    # Créer le logger root
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Supprimer les handlers existants
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Ajouter un handler pour la console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Ajouter un handler pour le fichier si spécifié
    if log_file:
        # Créer le répertoire si nécessaire
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Utiliser un TimedRotatingFileHandler pour rotation temporelle
        file_handler = TimedRotatingFileHandler(
            log_file, 
            when=when,
            interval=interval,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name, level=None):
    """
    Récupère un logger configuré avec un nom spécifique
    
    Args:
        name (str): Nom du logger
        level (int, optional): Niveau de journalisation
        
    Returns:
        logging.Logger: Logger configuré
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        # Convertir le niveau de log si c'est une chaîne
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        
        logger.setLevel(level)
    
    return logger

def add_file_handler(logger, log_file, level=logging.INFO, format_str=None, max_bytes=10485760, backup_count=5):
    """
    Ajoute un handler de fichier à un logger existant
    
    Args:
        logger (logging.Logger): Logger existant
        log_file (str): Chemin du fichier de log
        level (int, optional): Niveau de journalisation
        format_str (str, optional): Format des messages
        max_bytes (int, optional): Taille maximale du fichier de log
        backup_count (int, optional): Nombre de fichiers de sauvegarde
        
    Returns:
        logging.Logger: Logger mis à jour
    """
    # Format par défaut
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Créer le formatter
    formatter = logging.Formatter(format_str)
    
    # Créer le répertoire si nécessaire
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Utiliser un RotatingFileHandler pour limiter la taille des fichiers
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_bytes, 
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    
    # Ajouter le handler au logger
    logger.addHandler(file_handler)
    
    return logger


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration du logging
    logger = setup_logger(level=logging.DEBUG, log_file="logs/test.log")
    
    # Exemple d'utilisation
    logger.debug("Message de debug")
    logger.info("Message d'information")
    logger.warning("Message d'avertissement")
    logger.error("Message d'erreur")
    
    # Obtenir un logger avec un nom spécifique
    module_logger = get_logger("module_test")
    module_logger.info("Message du module")