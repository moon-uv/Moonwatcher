In this repository, we have:
- Short_Cadence_Kepler_light_curve_data.dat: A list of all stars from the Kepler mission for which Short Cadence data are available. This corresponds to Table S1 in the paper.
- Downloader.py: A program for downloading Short Cadence light curves from the Kepler mission via the MAST archive, using the LightKurve software package.
- Moonwatcher.py: An example program for analyzing Kepler light curves (downloaded using the aforementioned program) to search for signals of natural satellites orbiting these planets.
- exoplanet_data.tsv: An auxiliary file for the program above, containing data on various Kepler planets—such as orbital period, transit duration, and the masses of the star and the planet.
