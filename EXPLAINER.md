# Interview explainer

The final pipeline is deliberately staged: read the image, transform it to
colour spaces, create an egg-like foreground mask, remove small components,
and apply conservative morphology. Each threshold is in YAML so a change can
be tested as an experiment rather than hidden in code.

Task 1 uses SAM only to create pseudo-ground-truth masks. Those masks are
validated and then compared with classical predictions; SAM output is never an
input feature at inference time. The split groups Roboflow-derived filenames so
augmented copies do not cross the dev/eval boundary.

Live modification points: the colour-space thresholds, morphology kernel,
connected-component filtering, and the split grouping rule. Explain IoU as
intersection over union, Dice as overlap weighted toward foreground, and the
Hungarian algorithm as minimum-cost one-to-one association between detections
and existing tracks. Call out that pseudo-label errors and hand/glare/overlap
remain limitations.
