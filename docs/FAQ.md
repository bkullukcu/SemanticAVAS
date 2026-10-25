# Frequently asked questions

**Can I use my own semantic adjectives?**

Yes. Define each scale and its numerical range in the JSON schema. Keep the adjective meaning and score direction consistent throughout the study.

**How are repeated ratings handled?**

Declare a trial ID when participants rate a stimulus repeatedly. `SemanticAtlas` averages repeated trials within each participant–stimulus pair, then computes participant-balanced stimulus profiles.

**Which sounds are considered by target matching?**

The standard ranking evaluates the catalogue for which listener ratings are available. Digital synthesis offers additional candidate waveforms for a subsequent listening experiment.

**How are the intervals in the README calculated?**

The plotting script simulates 18 listener profiles and resamples listeners to produce pointwise bootstrap intervals. The accompanying synthetic example can be regenerated locally.

**Can ratings from different studies be compared?**

Yes, once the questionnaire constructs, scale ranges and score directions have been mapped and justified. Keep the original study definitions in the accompanying metadata.
