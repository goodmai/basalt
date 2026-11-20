# Basalt pour Mac

> Languages: [English](../../mac-app.md) · [Русский](../ru/mac-app.md) · [中文](../zh/mac-app.md) · [Español](../es/mac-app.md) · **Français** · [日本語](../ja/mac-app.md) · [Deutsch](../de/mac-app.md)

Une application native autour de basalt : une fenêtre de chat et une fenêtre Moniteur, reposant sur un moteur local que l'application démarre et supervise. La conception et les phases sont décrites dans #434. Il s'agit ici de la phase 1.

## Compilation et exécution

```bash
./scripts/bundle_app.swift            # release build, Metal kernels, then .build/Basalt.app, signed ad hoc
open .build/Basalt.app
```

- `--identity "Developer ID Application: …"` signe avec le runtime renforcé (hardened runtime) pour la distribution.
- `--bin PATH` assemble l'application à partir de binaires déjà compilés.
- `--dry-run` affiche le plan.

Depuis un dossier de compilation sans empaquetage, pointez l'application vers le moteur : `BASALT_HELPER=.build/release/Basalt .build/release/BasaltMac`.

## Fonctionnement

- **Le moteur est un processus distinct :** le binaire `basalt serve` empaqueté, dans `Contents/Helpers`.
  - L'application le démarre sur un port de boucle locale libre (`--port 0 --announce-ready`) et lui transmet un jeton par lancement dans `BASALT_API_KEY`.
  - Elle communique avec le moteur via la même API REST que celle utilisée par les éditeurs et les scripts.
- **Plantages :** un plantage du moteur n'entraîne pas celui de l'application. Le superviseur le redémarre après 1 s, puis 2 s et ainsi de suite jusqu'à 30 s, et abandonne après trois plantages en une minute en affichant la dernière sortie du moteur.
- **Fermeture :** quitter l'application arrête le moteur proprement : SIGINT, puis SIGKILL après 10 s.
- **Relance :** l'application adopte un moteur encore en cours d'exécution à partir de son fichier d'enregistrement (`~/Library/Application Support/Basalt/engine.json`, mode 0600), afin de ne pas recharger le modèle.
- **Les conversations** sont stockées avec SwiftData dans `~/Library/Application Support/Basalt/Conversations.store`. Une réponse interrompue par un plantage revient marquée comme arrêtée.
- **Le modèle** est défini dans les Réglages (par défaut `mlx-community/gemma-4-12B-it-qat-4bit`). Il s'applique au prochain démarrage du moteur.

## Barre de menus et requête rapide
- **Le moteur démarre avec l'application**, et non avec la fenêtre. L'icône de la barre de menus (un cube) est présente même lorsque toutes les fenêtres sont fermées. Elle affiche l'état et le modèle du moteur et propose les actions suivantes :
  - nouvelle conversation et requête rapide ;
  - les cinq conversations les plus récentes ;
  - démarrer, redémarrer et arrêter le moteur ;
  - le Moniteur, les Réglages et Quitter.
- **Requête rapide :** ⌥Espace ouvre un panneau flottant depuis n'importe quelle application, y compris en plein écran, sans lui retirer le focus.
  - Saisissez une question et appuyez sur ↩. La réponse s'affiche en streaming dans le panneau.
  - ⌘↩ ouvre l'échange en tant que conversation ordinaire dans la fenêtre principale.
  - Esc ferme le panneau et supprime l'échange. Tant que ⌘↩ n'est pas pressé, rien issu de la requête rapide n'est enregistré.
  - Le raccourci est enregistré avec `RegisterEventHotKey`, qui ne nécessite aucune autorisation d'accessibilité. Il peut être désactivé dans les Réglages ; ce changement s'applique au lancement suivant.
- **Réglages :**
  - Général : le modèle, la requête rapide et l'ouverture à la connexion (`SMAppService` ; fonctionne une fois l'application signée et placée dans Applications).
  - Clés Cloud : variables d'environnement, telles que `OPENROUTER_API_KEY`, dont les valeurs sont conservées dans le Trousseau (service `app.basalt.mac`). Le moteur les reçoit lors de son prochain démarrage pour les modèles cloud compose. L'application ne les écrit jamais sur disque.
- **Restauration d'état :** la conversation sélectionnée est restaurée après une relance. Il en va de même pour un brouillon non envoyé, par conversation, conservé dans les préférences utilisateur (user defaults).

## Modèles
- **La fenêtre Modèles** (⌥⌘L, ou Modèles dans la barre de menus) liste les modèles présents dans le cache Hugging Face (`~/.cache/huggingface/hub`) avec leur taille et un verdict de compatibilité mémoire pour ce Mac, fourni par `GET /v1/models/preview`.
- **Charger** bascule le moteur sur ce modèle (`POST /v1/models/load`) et en fait le modèle de démarrage du moteur. Un modèle dont le verdict est « incompatible » ("does not fit") ne peut pas être chargé depuis l'application ; la fenêtre en indique la raison.
- **Télécharger** prend un identifiant de dépôt tel que `mlx-community/gemma-4-12B-it-qat-4bit`. L'application télécharge dans son propre processus, affiche la progression et mémorise les téléchargements inachevés. Au lancement suivant, elle les reprend à partir des fichiers partiels. Un téléchargement échoué attend que vous cliquiez sur Réessayer.
- **Supprimer** demande d'abord confirmation, puis retire le dossier du modèle du cache. Le modèle actuellement chargé ne peut pas être supprimé.

## Pièces jointes
- **Joindre** avec le trombone ou en déposant des fichiers sur la conversation : PDF, images (PNG, JPEG, HEIC, TIFF, GIF, WebP) et audio (WAV, MP3, M4A, AAC, FLAC, CAF), jusqu'à 32 MB chacun.
- **Un message avec pièces jointes passe par `/v1/pipeline`.** Le moteur lit les documents et les images avec son modèle OCR, ou avec Apple Vision s'il n'en a pas. Il transcrit l'audio avec son modèle vocal, ou avec le modèle de chat. Le modèle de chat répond ensuite au message (ou « Résumez les fichiers joints. » lorsque le message est vide).
- **Le texte perçu reste dans la conversation.** Chaque pièce jointe apparaît sous forme d'étiquette (chip), qui révèle le texte lu par le moteur. Les messages suivants sont envoyés avec ce texte placé avant la question, afin que les relances puissent faire référence au fichier.
- **Le reçu** sous la réponse liste chaque étape du pipeline avec son modèle et sa durée en millisecondes.

## Inspecteur
- **⌥⌘I**, ou un clic sur une réponse, ouvre l'inspecteur à côté de la conversation.
- **Conversation :** le modèle et une invite système (system prompt) pour cette conversation. Elle est enregistrée avec la conversation et envoyée avant chaque message. Une invite vide l'efface.
- **Réponse :** le reçu de la réponse sélectionnée : modèle, jetons d'invite et de complétion, temps jusqu'au premier jeton, vitesse, ainsi que chaque étape du pipeline avec son modèle et sa durée en millisecondes.
- **Pièces jointes :** le texte lu par le moteur à partir de chaque fichier de la question, page par page.

## Raccourcis, Services et Dock
- **Raccourcis et Spotlight :** Ask Basalt, Summarize File, Transcribe Audio et Switch Model sont des App Intents. L'application Raccourcis les exécute avec l'application en arrière-plan. Chacune attend que le moteur soit prêt (jusqu'à trois minutes), et Switch Model refuse tout modèle qui ne tient pas en mémoire. `scripts/bundle_app.swift` extrait les métadonnées des intents dans `Contents/Resources/Metadata.appintents` à l'aide de `appintentsmetadataprocessor`, car SwiftPM ne le fait pas.
- **Services :** sélectionnez du texte dans n'importe quelle application et choisissez Services → Ask Basalt. Une nouvelle conversation s'ouvre avec ce texte comme question.
- **Lancement en un clic via `basalt://` depuis Hugging Face :** `Basalt.app` enregistre le schéma d'URL `basalt` (`CFBundleURLTypes`). L'ouverture de `basalt://run?model=<owner/repo>[&role=chat|mtp|moe-stream|ocr|asr|decide|embed|rerank]` — ou un clic sur **Run with Basalt** sur n'importe quelle fiche de modèle [`goodmai`](https://huggingface.co/goodmai) — valide l'identifiant du dépôt Hugging Face, met à jour le choix de modèle enregistré, met Basalt au premier plan et bascule le moteur supervisé vers ce modèle ou ce rôle de pipeline.
- **Dock :** un fichier PDF, image ou audio déposé sur l'icône du Dock démarre une nouvelle conversation avec ce fichier en pièce jointe.
- **Notifications :** lorsqu'une réponse se termine alors que Basalt est en arrière-plan, une notification affiche ses premières lignes. L'autorisation n'est demandée qu'une seule fois.

## Systèmes Compose et cloud
- **Réglages → Moteur :** exécutez un modèle unique ou un système compose à partir d'un fichier (`basalt serve --compose <file>`). « Redémarrer le moteur avec ceci » arrête le moteur et le démarre avec les nouveaux arguments. Ce choix est également conservé au lancement suivant.
- **L'adoption se fait par les arguments :** l'application n'adopte un moteur laissé actif que s'il a été démarré avec les mêmes arguments. Après un changement de modèle ou de fichier compose, un nouveau moteur démarre. Quitter l'application arrête également un moteur adopté.
- **Moniteur :** pour un système compose, affiche son nom, ses pipelines et ses dépenses cloud du jour, ainsi que pour chaque modèle son état (`ready`, `loading`, `no-key`, `over-budget`, etc.), sa politique de conservation, ses dépenses et son plafond, ainsi que tout refus éventuel. Les modèles cloud portent un symbole de nuage.
- **Le marqueur cloud :** une réponse ayant utilisé un modèle cloud indique « ☁︎ cloud » avec son coût sur la ligne de reçu et dans l'inspecteur. Tout ce qui ne porte pas ce marqueur s'est exécuté localement sur ce Mac.

## Accessibilité, langues et mises à jour
- **VoiceOver :**
  - Chaque commande représentée uniquement par une icône possède un nom vocalisé : l'icône de la barre de menus, envoyer, arrêter, joindre, supprimer et l'inspecteur.
  - Chaque message est lu comme un seul élément, « Vous : … » ("You: …") ou « Basalt : … » ("Basalt: …").
  - Les indicateurs de progression précisent ce qu'ils attendent.
  - Le badge de compatibilité indique « convient à ce Mac » ("fits on this Mac") ou « ne convient pas » ("does not fit").
  - Les pièces jointes en attente sont lues comme « Supprimer <file> » ("Remove <file>").
- **Langues :** l'application suit la langue du système. L'anglais et le russe sont inclus dès maintenant. Une traduction correspond à `Localization/<language>.lproj/Localizable.strings`, que `scripts/bundle_app.swift` copie dans le bundle et liste dans `CFBundleLocalizations`. Un test échoue lorsqu'une vue ajoute une chaîne absente de la table russe.
- **Mises à jour :** une fois par jour, l'application interroge l'API GitHub pour connaître la dernière version de basalt. Une version plus récente, qui n'est ni un brouillon ni une pré-version, s'affiche comme « Mise à jour disponible » ("Update available") dans la barre de menus, avec un lien vers la page de la release. La requête n'envoie rien d'autre à GitHub que l'adresse IP et le user-agent de l'application. Désactivez cette option dans Réglages → Général.

## Distribution : signature, notarisation et cask Homebrew
- **En local :** `./scripts/bundle_app.swift --identity "Developer ID Application: Name (TEAMID)" --notarize basalt-notary --dmg` signe avec le runtime renforcé et écrit `Basalt-<version>.zip`. Le script soumet l'archive zip avec `notarytool`, agrafe le ticket à l'application et la compresse de nouveau en zip, puis crée un `.dmg`.
  - `--zip` seul empaquette sans notariser.
  - `--notarize` refuse une signature ad hoc ou absente.
  - Créez le profil du trousseau une fois avec `xcrun notarytool store-credentials basalt-notary --apple-id <id> --team-id <TEAMID> --password <app-specific password>`.
- **Lors d'une release :** lorsque le dépôt dispose des secrets `MAC_SIGNING_IDENTITY` (le nom du certificat dans le trousseau du runner) et `MAC_NOTARY_PROFILE` (ce profil de trousseau), un tag `v*` compile, signe et notarise l'application.
  - Il joint `Basalt-<version>.zip` à la release GitHub.
  - Il génère `Casks/basalt-app.rb` à partir de `Casks/basalt-app.rb.template` dans la même PR qui met à jour la formule.
  - Sans ces secrets, la release ne publie que la formule, accompagnée d'un avis.
- **Installation :** dès qu'une release inclut l'application, exécutez `brew install --cask goodmai/basalt/basalt-app`.

## Compose Video Studio, Convertisseur universel & Live Hub (`v0.12`)
- **Compose Video Studio :** Génération vidéo par phases sur 24 GB UMA et lecture intégrée (`LTX-2.5 AV-DiT`, `Kandinsky 6.0`, `LingBot-Video`, `Wan 2.2`) avec export `.mp4` (`#611`).
- **Convertisseur universel (`ConvertModelSheet`) :** Interface complète de quantification tout-vers-tout (`MACaQ` dense/MoE, `FlatQuant`, `LIRQuant`, `SINQ`, `QTIP`, `E8P`, `PV-Tuning`, `QAT STE` + `QuantizationAuditCore`) en anglais et en russe (`#546`, `#599`, `#623`).
- **Live Hub & télémétrie SSE 10 Hz :** Propulsé par `swift-huggingface` avec téléchargements Xet, recherche/publication Hub en direct et télémétrie en streaming à 10 Hz dans le Moniteur (`#587`).

## Budgets
Mesuré sur un M4 Pro avec 24 GB au commit 530aa28, avec Gemma 4 12B dans le moteur (#434) :

| Budget | Mesuré | Limite | Méthode |
|---|---|---|---|
| Fenêtre au lancement à chaud | 257 ms médiane sur 5 (à froid : 1.6 s) | 400 ms | `swift scripts/app_budgets.swift` |
| Réveils au repos une fois le moteur démarré | 0 en 10 s | 0 | idem |
| Processus de l'application | 40 MB, plus 33.5 MB pour une transcription ouverte de 2,000 messages | 150 MB | idem, et `AppBudgetTests` |
| Un jeton diffusé dans une transcription de 2,000 messages | 7.4 ms p95, 6.0 ms médiane | 16.7 ms (60 fps) | `BASALT_APP_BUDGETS=1 swift test --filter AppBudgetTests` |
| Requête rapide : panneau et premier jeton au-delà du temps propre du moteur | 40 ms à chaud (panneau 15 ms) ; première vue SwiftUI d'un processus 175 ms | 150 ms | idem |

`app_budgets.swift` prend le verrou GPU, puisque le moteur charge son modèle. Exécutez les deux sur un Mac calme, sans tâche CI en cours. Trois points demandent une vérification manuelle non automatisée : le panneau au-dessus d'une application plein écran (c'est un panneau non activant avec `.fullScreenAuxiliary`), les App Intents depuis Raccourcis lorsque l'application est fermée, et une installation par cask sur un Mac vierge, qui nécessite les secrets de signature.
