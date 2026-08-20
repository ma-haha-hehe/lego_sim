# Third-party and release notice

This repository is currently a research workspace, not yet a legally curated
binary distribution.

- The Franka Panda MJCF files under `src/mj_bridge/mj_bridge` retain their
  Apache-2.0 license.
- FoundationPose is distributed under NVIDIA's license and carries a
  non-commercial research/evaluation use limitation. Its license must remain
  with any permitted redistribution.
- The IFM O3P SDK directory contains vendor binaries and documentation. Confirm
  the vendor redistribution terms or exclude it from public source releases.
- Two legacy LEGO/Duplo mesh files in the research workspace lack provenance
  metadata. Public episodes use self-authored box/cylinder primitives instead;
  `scripts/create_public_release.sh` excludes those legacy meshes.
- Model weights, Docker/container home directories, camera captures, build
  outputs and real-robot logs should not be included in a minimal benchmark
  release.

The word LEGO is a trademark of the LEGO Group, which does not sponsor,
authorize, or endorse this research environment.
