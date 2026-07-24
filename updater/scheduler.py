"""
Planification automatique des vérifications de mises à jour.

Utilise APScheduler pour déclencher updater.reindex.run_full_update_cycle()
à intervalles réguliers (par défaut toutes les 24h), sans intervention
manuelle. Ce script est prévu pour tourner en arrière-plan pendant que
l'application Chainlit est active.
"""

import os
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler

from updater.reindex import run_full_update_cycle
from updater.monitor import log_event

load_dotenv()

UPDATE_CHECK_INTERVAL_HOURS = int(os.getenv("UPDATE_CHECK_INTERVAL_HOURS", "24"))


def scheduled_job():
    """Tâche exécutée à chaque intervalle : log de début, cycle complet, log de fin."""
    log_event("=== Déclenchement de la vérification planifiée ===")
    try:
        run_full_update_cycle()
    except Exception as e:
        log_event(f"ERREUR lors du cycle planifié : {e}")


def start_scheduler() -> BackgroundScheduler:
    """
    Démarre le scheduler en arrière-plan. Ne bloque pas l'exécution du
    reste du programme (c'est important : Chainlit doit pouvoir tourner
    en même temps que cette surveillance).
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_job,
        trigger="interval",
        hours=UPDATE_CHECK_INTERVAL_HOURS,
        id="force_n_update_check",
        next_run_time=None,  # ne se déclenche pas immédiatement au démarrage
    )
    scheduler.start()
    print(f"Scheduler démarré : vérification toutes les {UPDATE_CHECK_INTERVAL_HOURS}h.")
    return scheduler


if __name__ == "__main__":
    # Test manuel : on déclenche un cycle immédiatement pour valider que
    # tout fonctionne, sans attendre l'intervalle complet.
    print("Test manuel du scheduler : exécution immédiate d'un cycle...")
    scheduled_job()

    print("\nDémarrage du scheduler en arrière-plan (Ctrl+C pour arrêter)...")
    scheduler = start_scheduler()

    try:
        import time
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        print("Scheduler arrêté proprement.")