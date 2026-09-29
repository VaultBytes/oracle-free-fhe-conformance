# Companion note

**What a Metamorphic Relation Can and Cannot See in an FHE Decoder**
Bader Alissaei, 2026. Four pages.

Cited by *Decoded-Output Conformance Need Not Establish CKKS Execution* as `[degreenote]`. It is
here so that citation resolves.

The note gives a criterion for which decoder faults a relational test can see. A relation assembled
from device decodes is blind to a multiplicative decoder error exactly when it is invariant under
`D -> sD`, and blind to an additive one exactly when its coefficients sum to zero. Both conditions
are checked against the suite in this repository, where all three laws turn out to be blind to a
scale error and two of the three blind to an offset.

Source and PDF are both here. The measurements in it were taken against this repository.
