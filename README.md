# USC8 AI Workshop

Questa repository contiene il materiale didattico (notebook Jupyter) preparato per il workshop sull'Intelligenza Artificiale organizzato da USC8. I notebook coprono diverse applicazioni di Machine Learning e Deep Learning in ambito astronomico.

## Contenuti

La repository include i seguenti notebook principali:

*   **`redshift_regression.ipynb`**:
    *   **Obiettivo**: Prevedere il redshift fotometrico (`z_phot`) di oggetti astronomici.
    *   **Dataset**: Utilizza dati tabellari (magnitudini, errori, raggi, ecc.) provenienti da una query SDSS (Sloan Digital Sky Survey), caricati da un file FITS (`catania_cavuoti.fit`).
    *   **Tecniche**: Esplora diversi algoritmi di regressione da Scikit-learn (Random Forest, K-Nearest Neighbors, Multi-Layer Perceptron). Include pre-processing dei dati (scaling), feature importance analysis (con Random Forest), e ottimizzazione degli iperparametri (GridSearch).
    *   **Librerie Principali**: `scikit-learn`, `pandas`, `numpy`, `astropy`, `matplotlib`, `seaborn`.

*   **`gmnist_classifier_light.ipynb`**:
    *   **Obiettivo**: Classificare immagini di galassie in base alla loro morfologia (4 classi: smooth_round, smooth_cigar, edge_on_disk, unbarred_spiral).
    *   **Dataset**: Utilizza il dataset Galaxy MNIST (immagini PNG in 3 bande ottiche *grz* da DECaLS/Galaxy Zoo).
    *   **Tecniche**: Implementa classificatori CNN (Convolutional Neural Network) usando PyTorch e Torchvision. Mostra come usare un'architettura pre-addestrata (ResNet) e come definire una CNN custom. Include data augmentation, valutazione del modello, visualizzazione delle feature map e delle mappe di attivazione (Grad-CAM).
    *   **Librerie Principali**: `pytorch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `PIL`, `opencv-python`, `pytorch-grad-cam`.

*   **`plasticc_classifier.ipynb`**:
    *   **Obiettivo**: Classificare curve di luce (serie temporali) di transienti astronomici (15 classi).
    *   **Dataset**: Utilizza il dataset PLAsTiCC (Photometric LSST Astronomical Time Series Classification Challenge), con dati simulati simili a quelli di LSST. I dati sono in formato CSV (metadata e lightcurves) scaricati da Zenodo.
    *   **Tecniche**: Costruisce un classificatore basato su RNN (Recurrent Neural Network, specificamente GRU) usando PyTorch. Include la definizione di un `Dataset` custom, gestione di sequenze di lunghezza variabile (padding), data augmentation specifica per time-series, e l'implementazione di una metrica di loss custom (weighted multi-class log-loss) adatta alla challenge PLAsTiCC.
    *   **Librerie Principali**: `pytorch`, `pandas`, `numpy`, `scikit-learn` (per split), `matplotlib`, `tqdm`.

Altri file:
*   **`.gitignore`**: File standard che indica a Git quali file o cartelle ignorare.
*   **`LICENSE`**: Il file che specifica la licenza sotto cui è distribuito il materiale (presumibilmente MIT License).

## Come Eseguire i Notebook in Google Colab (Metodo Consigliato)

Il modo più semplice per interagire con questi notebook è utilizzare Google Colaboratory (Colab), un servizio gratuito che permette di eseguire codice Python direttamente nel browser, senza bisogno di installazioni locali e fornendo accesso a risorse computazionali (come GPU gratuite, utili specialmente per i notebook basati su PyTorch).

**Opzione 1: Link Diretti (Badge "Open in Colab")**

Clicca sui badge qui sotto per aprire ciascun notebook direttamente in Google Colab:

*   **Redshift Regression:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simoneriggi/usc8-ai-workshop/blob/main/redshift_regression.ipynb)
*   **Galaxy MNIST Classifier:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simoneriggi/usc8-ai-workshop/blob/main/gmnist_classifier_light.ipynb)
*   **PLAsTiCC Classifier:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simoneriggi/usc8-ai-workshop/blob/main/plasticc_classifier.ipynb)

**Opzione 2: Apertura Manuale da Colab**

1.  Apri [Google Colab](https://colab.research.google.com/).
2.  Vai su `File` -> `Apri notebook`.
3.  Seleziona la scheda `GitHub`.
4.  Incolla l'URL di questa repository: `https://github.com/simoneriggi/usc8-ai-workshop`
5.  Premi Invio o clicca sull'icona di ricerca.
6.  Dovrebbe apparire la lista dei file. Clicca sul notebook `.ipynb` che desideri aprire.

Una volta aperto in Colab, puoi eseguire le celle di codice una ad una premendo `Shift + Invio` o usando i pulsanti nell'interfaccia. I notebook contengono celle per installare le librerie necessarie (`%pip install ...`), quindi dovrebbero funzionare direttamente in Colab.

Ricorda che l'ambiente Colab è temporaneo; se fai modifiche che vuoi salvare, assicurati di farlo (`File -> Salva una copia su Drive` o `File -> Scarica .ipynb`).

## Prerequisiti

*   Per usare Google Colab: Un account Google.
*   Una conoscenza di base di Python e dei concetti fondamentali di Machine Learning è utile per seguire i notebook. Le librerie specifiche vengono installate direttamente all'interno dei notebook stessi.

## Licenza

Il contenuto di questa repository è rilasciato sotto la licenza MIT. Per maggiori dettagli, consulta il file `LICENSE`.

---

Spero questo workshop sia utile e interessante!