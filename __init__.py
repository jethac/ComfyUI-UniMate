"""ComfyUI UniMate entry point; subsystem tests can import without ComfyUI."""


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension, io
    from .nodes import (
        UniMateExportGLB,
        UniMateGenerateMotion,
        UniMateLoadRig,
        UniMateModelLoader,
        UniMatePrepareRig,
        UniMateInbetweenMotion,
        UniMateEditMotion,
        UniMateLoadMotion,
        UniMateSaveMotion,
        UniMateExpandMotion,
        UniMateExtractMotion,
        UniMateGenerateBatch,
        UniMateCombineRigs,
        UniMateCanonicalAsset,
        UniMateRigConditioning,
        UniMateExportFBX,
        UniMateRecoverSkeleton,
        UniMatePreviewSkeleton,
        UniMateFootLockMotion,
        UniMateBuildDataset,
        UniMateSplitDataset,
        UniMatePlanSampling,
        UniMateLoadDataset,
        UniMateSaveDataset,
        UniMateDatasetStatistics,
        UniMateLoadStatistics,
        UniMateSaveStatistics,
    )

    class UniMateExtension(ComfyExtension):
        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            return [
                UniMateLoadRig,
                UniMatePrepareRig,
                UniMateModelLoader,
                UniMateGenerateMotion,
                UniMateExportGLB,
                UniMateInbetweenMotion,
                UniMateEditMotion,
                UniMateLoadMotion,
                UniMateSaveMotion,
                UniMateExpandMotion,
                UniMateExtractMotion,
                UniMateGenerateBatch,
                UniMateCombineRigs,
                UniMateCanonicalAsset,
                UniMateRigConditioning,
                UniMateExportFBX,
                UniMateRecoverSkeleton,
                UniMatePreviewSkeleton,
                UniMateFootLockMotion,
                UniMateBuildDataset,
                UniMateLoadDataset,
                UniMateSaveDataset,
                UniMateDatasetStatistics,
                UniMateLoadStatistics,
                UniMateSaveStatistics,
                UniMateSplitDataset,
                UniMatePlanSampling,
            ]

    return UniMateExtension()
