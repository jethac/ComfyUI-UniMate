# Build from a Cloud Offload runner containing this task's coordinator changes.
ARG CLOUD_OFFLOAD_IMAGE
FROM ${CLOUD_OFFLOAD_IMAGE}

USER root
RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl xz-utils libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 libsm6 libegl1 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --retry 3 \
      https://download.blender.org/release/Blender5.1/blender-5.1.1-linux-x64.tar.xz \
      --output /tmp/blender.tar.xz \
    && echo '6f9fff89fef154ef7974d1a1c4b916ab4bc1f5618bcb48d5befee1bd0a7c7f2a  /tmp/blender.tar.xz' | sha256sum --check \
    && mkdir /opt/blender \
    && tar --extract --xz --file /tmp/blender.tar.xz --directory /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz

COPY . /opt/ComfyUI/custom_nodes/comfy-unimate
RUN python -m pip install --no-cache-dir -r /opt/ComfyUI/custom_nodes/comfy-unimate/requirements.txt \
    && /opt/blender/blender --background --factory-startup --disable-autoexec \
         --python-expr 'import bpy, numpy; assert bpy.app.version[:2] == (5, 1)'

ENV UNIMATE_BLENDER=/opt/blender/blender \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Preserve the base runner's entrypoint and capability manifest. Models are staged.
WORKDIR /opt/ComfyUI
