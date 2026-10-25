# FAQ

**Can I import a questionnaire with different semantic adjectives?**

Yes. List the scales and their numerical limits in the input JSON schema. The software does not guess whether a scale should be inverted or whether two different questionnaires measure the same construct.

**What happens to repeated ratings from one listener?**

Repeated trials require explicit trial IDs during import. `SemanticAtlas` first averages repeats within participant and stimulus, then averages participant profiles. A listener with extra trials should not receive extra weight solely for that reason.

**Can the package choose a sound no one has evaluated?**

The standard `rank` method ranks existing, rated sounds. The synthesis helper produces a digital waveform, but its subjective ratings are unknown until tested with listeners.

**Do the plotted confidence intervals describe real listeners?**

No. The README uses 18 simulated listener profiles. The intervals illustrate participant-resampling, not a confidence statement about any study population.

**Can I combine the AVAS scales with ARAUS soundscape ratings?**

Not directly. Their constructs and ranges differ. Any harmonization must be justified and validated outside the default import workflow.
