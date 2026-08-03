# SquidXplorer

Post-acquisition viewing and processing of finished Squid high-content-screening scans. The
vocabulary below is the one the code should use; where the code currently disagrees, the code is
what needs changing.

## Language

**Acquisition**:
One finished Squid scan as it sits on disk, with its images and the metadata that places them.
_Avoid_: dataset, experiment, run

**Region**:
The unit an operator runs over: one contiguous imaged area of an acquisition. This is the general
term. A region may contain many FOVs.
_Avoid_: site, area, tile

**Well**:
A region on a plate acquisition, named by its row and column. Every well is a region; not every
region is a well, because a glass-slide acquisition has regions and no wells.
_Avoid_: using this interchangeably with region

**FOV**:
One camera frame position within a region. A region of 27 FOVs is 27 overlapping frames, not one
frame stretched over the region.
_Avoid_: tile, position. _Field_ is NGFF's spelling and is correct only when describing the
on-disk NGFF structure, never when describing an acquisition.

**Mosaic**:
A region's FOVs composited into one image at their stage coordinates.
_Avoid_: stitched image (stitching is one particular way to produce a mosaic, with registration)

**Plane**:
The single two-dimensional image at one region, FOV, channel, z and timepoint. The unit that is
read from disk.
_Avoid_: frame, slice, image

**Operator**:
A named transform applied over an acquisition, such as MIP, reference plane, or stitch.
_Avoid_: operation, processor, job

**Operator run**:
One execution of an operator over a stated scope of regions.
_Avoid_: job, task, batch

**Preview**:
The raw imagery painted when an acquisition opens, before any operator has run. A preview is not a
processed result and never becomes one.
_Avoid_: thumbnail, overview

**First paint**:
The interval from a user asking for something to the first pixels of it being drawn on screen: from
starting an operator run to that run's first tile, and from asking for a window to that window's
first mosaic layer. Distinct in both cases from the total duration, and taken where the drawing
happens rather than where the producer emits it, because the difference between those two is queue
delay.
_Avoid_: latency, startup time, time to first byte
