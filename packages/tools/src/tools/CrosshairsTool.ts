import { vec2, vec3 } from 'gl-matrix';
import vtkMath from '@kitware/vtk.js/Common/Core/Math';
import vtkMatrixBuilder from '@kitware/vtk.js/Common/Core/MatrixBuilder';

import { AnnotationTool } from './base';

import { getRenderingEngine, type Types } from '@cornerstonejs/core';
import {
  getEnabledElementByIds,
  getEnabledElement,
  utilities as csUtils,
  Enums,
  CONSTANTS,
  triggerEvent,
  eventTarget,
  StackViewport,
  VolumeViewport3D,
} from '@cornerstonejs/core';

import {
  getToolGroup,
  getToolGroupForViewport,
} from '../store/ToolGroupManager';

import {
  addAnnotation,
  getAnnotations,
  removeAnnotation,
} from '../stateManagement/annotation/annotationState';

import {
  drawCircle as drawCircleSvg,
  drawHandles as drawHandlesSvg,
  drawLine as drawLineSvg,
} from '../drawingSvg';
import { state } from '../store/state';
import { Events } from '../enums';
import { getViewportIdsWithToolToRender } from '../utilities/viewportFilters';
import getViewportICamera from '../utilities/getViewportICamera';
import {
  resetElementCursor,
  hideElementCursor,
} from '../cursors/elementCursor';
import liangBarksyClip from '../utilities/math/vec2/liangBarksyClip';
import {
  getSlabThicknessOrDefault,
  jumpToFocalPoint,
} from '../utilities/genericViewportToolHelpers';
import {
  getWorldPointManager,
  type CameraPlane,
} from './WorldPointManager';

import * as lineSegment from '../utilities/math/line';
import type {
  Annotation,
  AnnotationData,
  Annotations,
  EventTypes,
  ToolHandle,
  PublicToolProps,
  ToolProps,
  InteractionTypes,
  SVGDrawingHelper,
} from '../types';
import { isAnnotationLocked } from '../stateManagement/annotation/annotationLocking';
import triggerAnnotationRenderForViewportIds from '../utilities/triggerAnnotationRenderForViewportIds';

const { RENDERING_DEFAULTS } = CONSTANTS;

/**
 * Displayed canvas size for crosshairs geometry. Native viewports use the
 * element size (their `viewport.canvas` is hidden -> 0x0); legacy uses the
 * canvas size.
 */
function getDisplayedCanvasSize(viewport): {
  clientWidth: number;
  clientHeight: number;
} {
  if (csUtils.isGenericViewport(viewport)) {
    const { element } = viewport;
    return {
      clientWidth: element.clientWidth,
      clientHeight: element.clientHeight,
    };
  }
  const { clientWidth, clientHeight } = viewport.canvas;
  return { clientWidth, clientHeight };
}

export type CrosshairsAnnotationData = AnnotationData & {
  handles: {
    rotationPoints: Types.Point3[]; // rotation handles, used for rotation interactions
    slabThicknessPoints: Types.Point3[]; // slab thickness handles, used for setting the slab thickness
    activeOperation: number | null; // 0 translation, 1 rotation handles, 2 slab thickness handles
    toolCenter: Types.Point3;
  };
  activeViewportIds: string[]; // a list of the viewport ids connected to the reference lines being translated
  viewportId: string;
};

export type CrosshairsAnnotation = Annotation & {
  data: CrosshairsAnnotationData;
};

function defaultReferenceLineColor() {
  return 'rgb(0, 200, 0)';
}

function defaultReferenceLineControllable() {
  return true;
}

function defaultReferenceLineDraggableRotatable() {
  return true;
}

function defaultReferenceLineSlabThicknessControlsOn() {
  return true;
}

const OPERATION = {
  DRAG: 1,
  ROTATE: 2,
  SLAB: 3,
};

/**
 * CrosshairsTool is a tool that provides reference lines between different viewports
 * of a toolGroup. Using crosshairs, you can jump to a specific location in one
 * viewport and the rest of the viewports in the toolGroup will be aligned to that location.
 * Crosshairs have grababble handles that can be used to rotate and translate the
 * reference lines. They can also be used to set the slab thickness of the viewports
 * by modifying the slab thickness handles.
 *
 */
class CrosshairsTool extends AnnotationTool {
  static toolName;
  static minimalModeExamples = new Map<
    string,
    {
      enabled: boolean;
      lineLengthInPx: number;
    }
  >([
    ['Default', { enabled: false, lineLengthInPx: 40 }],
    ['Minimal 40px', { enabled: true, lineLengthInPx: 40 }],
    ['Minimal 80px', { enabled: true, lineLengthInPx: 80 }],
  ]);

  _localToolCenter: Types.Point3 = [0, 0, 0];
  get toolCenter(): Types.Point3 {
    const forUID = this._getOwnFrameOfReferenceUID();
    if (forUID) {
      const manager = getWorldPointManager();
      const point = manager.getPoint(forUID);
      if (point) {
        return point;
      }
    }
    return this._localToolCenter;
  }
  set toolCenter(point: Types.Point3) {
    this._localToolCenter = [...point] as Types.Point3;
  }
  _lastValidToolCenter: Types.Point3 | null = null;
  _lastKnownFrameOfReferenceUID: string | null = null;
  _getReferenceLineColor?: (viewportId: string) => string;
  _getReferenceLineControllable?: (viewportId: string) => boolean;
  _getReferenceLineDraggableRotatable?: (viewportId: string) => boolean;
  _getReferenceLineSlabThicknessControlsOn?: (viewportId: string) => boolean;
  _volumeViewportNewVolumeListeners = new Map<
    string,
    {
      element: HTMLDivElement;
      handler: EventListener;
    }
  >();
  _toolGroupViewportAddedListener: EventListener | null = null;
  _toolGroupViewportRemovedListener: EventListener | null = null;
  _elementEnabledListener: EventListener | null = null;
  _imageRenderedListener: EventListener | null = null;
  _ignoreFiredEvents = false;
  // Raw STACK_NEW_IMAGE listeners (one per StackViewport) to detect slice/scroll changes.
  _stackImageListeners: Map<
    string,
    { element: HTMLDivElement; listener: EventListener }
  > = new Map();
  // StackViewports for which at least one real (loaded) image has been processed.
  _seenStackViewports: Set<string> = new Set();
  // Slice index last programmatically requested via setImageIdIndex() per
  // StackViewport; used to recognize and ignore the resulting async
  // STACK_NEW_IMAGE as our own doing (feedback-loop guard).
  _pendingProgrammaticSliceIndex: Map<string, number> = new Map();
  // Debounce handle: coalesces rapid TOOLGROUP_VIEWPORT_ADDED events (e.g. MPR
  // triad) into a single _computeToolCenter call once cameras have settled.
  _debouncedRecomputeHandle: ReturnType<typeof setTimeout> | null = null;
  _debounceDelayMs = 150;

  constructor(
    toolProps: PublicToolProps = {},
    defaultToolProps: ToolProps = {
      supportedInteractionTypes: ['Mouse'],
      configuration: {
        shadow: true,
        // renders a colored circle on top right of the viewports whose color
        // matches the color of the reference line
        viewportIndicators: false,

        viewportIndicatorsConfig: {
          radius: 5,
          x: null,
          y: null,
        },
        // Pans other viewports in the toolGroup when the crosshairs center
        // moves outside their bounds (e.g. while scrolling a zoomed view).
        autoPan: {
          enabled: false,
          panSize: 10,
        },
        handleRadius: 3,
        // Enable HDPI rendering for handles using devicePixelRatio
        enableHDPIHandles: false,
        // radius of the area around the intersection of the planes, in which
        // the reference lines will not be rendered. This is only used when
        // having 3 viewports in the toolGroup.
        referenceLinesCenterGapRadius: 20,
        // The ratio is a fraction of the minimum canvas dimension (width or height).
        // For example, if referenceLinesCenterGapRatio is set to 0.05, the gap will be 5% of the smallest side of the canvas.
        // If set to 1, the gap will be equal to the minimum canvas dimension (which would likely hide the crosshairs).
        // referenceLinesCenterGapRatio: null|undefined → gap is referenceLinesCenterGapRadius (default: 20 pixels)
        // referenceLinesCenterGapRatio: 0.05 → gap is 5% of the canvas min dimension
        // referenceLinesCenterGapRatio: 0.1 → gap is 10% of the canvas min dimension
        // referenceLinesCenterGapRatio: 1 → gap is 100% (not recommended)
        referenceLinesCenterGapRatio: null,
        // Minimal crosshair mode renders a short fixed-length cross centered on
        // the tool center and disables rotation and slab thickness handles.
        minimal: {
          enabled: false,
          lineLengthInPx: 40,
        },
        // actorUIDs for slabThickness application, if not defined, the slab thickness
        // will be applied to all actors of the viewport
        filterActorUIDsToSetSlabThickness: [],
        // blend mode for slabThickness modifications
        slabThicknessBlendMode: Enums.BlendModes.MAXIMUM_INTENSITY_BLEND,
        centerPoint: {
          enabled: false,
          color: 'rgba(255, 255, 0, 0.5)',
          size: 2,
        },
        // Grabbable circle handle at the tool center for translating the whole
        // crosshair (all linked viewports). Works in Active and Passive modes.
        centerHandle: {
          enabled: true,
          // Visual + hit-test radius (canvas px). Falls back to `handleRadius`.
          radius: null,
          color: null,
        },
        mobile: {
          enabled: false,
          opacity: 0.8,
          handleRadius: 9,
          referenceLinesCenterGapRatio: 0.05,
        },
        // Emits verbose console logs for tool-center resolution/sync.
        // Can also be enabled globally via `globalThis.__CROSSHAIRS_DEBUG__ = true`.
        debug: false,
      },
    }
  ) {
    super(toolProps, defaultToolProps);

    this._getReferenceLineColor =
      toolProps.configuration?.getReferenceLineColor ||
      defaultReferenceLineColor;
    this._getReferenceLineControllable =
      toolProps.configuration?.getReferenceLineControllable ||
      defaultReferenceLineControllable;
    this._getReferenceLineDraggableRotatable =
      toolProps.configuration?.getReferenceLineDraggableRotatable ||
      defaultReferenceLineDraggableRotatable;
    this._getReferenceLineSlabThicknessControlsOn =
      toolProps.configuration?.getReferenceLineSlabThicknessControlsOn ||
      defaultReferenceLineSlabThicknessControlsOn;
  }

  /**
   * Gets the camera from the viewport, and adds crosshairs annotation for the viewport
   * to the annotationManager. If any annotation is found in the annotationManager, it
   * overwrites it.
   * @param viewportInfo - The viewportInfo for the viewport to add the crosshairs
   * @returns viewPlaneNormal and center of viewport canvas in world space
   */
  initializeViewport = ({
    renderingEngineId,
    viewportId,
  }: Types.IViewportId): {
    normal: Types.Point3;
    point: Types.Point3;
  } => {
    const enabledElement = getEnabledElementByIds(
      viewportId,
      renderingEngineId
    );
    if (!enabledElement) {
      return;
    }
    const { FrameOfReferenceUID, viewport } = enabledElement;
    const { element } = viewport;
    const { position, focalPoint, viewPlaneNormal } =
      getViewportICamera(viewport);

    // Check if there is already annotation for this viewport
    let annotations = this._getAnnotations(enabledElement);
    annotations = this.filterInteractableAnnotationsForElement(
      element,
      annotations
    );

    if (annotations?.length) {
      // If found, it will override it by removing the annotation and adding it later
      removeAnnotation(annotations[0].annotationUID);
    }

    const annotation = {
      highlighted: false,
      metadata: {
        cameraPosition: <Types.Point3>[...position],
        cameraFocalPoint: <Types.Point3>[...focalPoint],
        FrameOfReferenceUID,
        toolName: this.getToolName(),
      },
      data: {
        handles: {
          rotationPoints: [], // rotation handles, used for rotation interactions
          slabThicknessPoints: [], // slab thickness handles, used for setting the slab thickness
          toolCenter: this.toolCenter,
        },
        activeOperation: null, // 0 translation, 1 rotation handles, 2 slab thickness handles
        activeViewportIds: [], // a list of the viewport ids connected to the reference lines being translated
        viewportId,
      },
    };

    addAnnotation(annotation, element);

    this._registerViewportWithManager(renderingEngineId, viewportId);

    const { clientWidth, clientHeight } = getDisplayedCanvasSize(viewport);
    return {
      normal: viewPlaneNormal,
      point: viewport.canvasToWorld([clientWidth / 2, clientHeight / 2]),
    };
  };

  _getViewportsInfo = () => {
    const viewports = getToolGroup(this.toolGroupId).viewportsInfo;

    return viewports;
  };

  _reinitializeListenersAndCenter = (): void => {
    this._unbindToolGroupViewportListeners();
    this._clearAllVolumeListenersAndViewportState();
    this._bindToolGroupViewportListeners();
    this._registerAllViewportsWithManager();
    this._syncVolumeListenersWithToolGroup();
    this._computeToolCenter(this._getViewportsInfo());
  };

  onSetToolActive() {
    this._reinitializeListenersAndCenter();
  }

  onSetToolPassive() {
    this._reinitializeListenersAndCenter();
  }

  onSetToolEnabled() {
    this._reinitializeListenersAndCenter();
  }

  onSetToolDisabled() {
    const viewportsInfo = this._getViewportsInfo();

    this._unregisterAllViewportsFromManager();
    this._unbindToolGroupViewportListeners();
    this._clearAllVolumeListenersAndViewportState();
    this._unbindStackImageListeners();
    this._cancelDebouncedRecompute();
    this._ignoreFiredEvents = false;
    this.editData = null;
    state.isInteractingWithTool = false;

    // Crosshairs annotations in the state
    // has no value when the tool is disabled
    // since viewports can change (zoom, pan, scroll)
    // between disabled and enabled/active states.
    // so we just remove the annotations from the state
    viewportsInfo.forEach(({ renderingEngineId, viewportId }) => {
      const enabledElement = getEnabledElementByIds(
        viewportId,
        renderingEngineId
      );

      if (!enabledElement) {
        return;
      }

      const annotations = this._getAnnotations(enabledElement);

      if (annotations?.length) {
        annotations.forEach((annotation) => {
          removeAnnotation(annotation.annotationUID);
        });
      }
    });
  }

  resetCrosshairs = () => {
    const viewportsInfo = this._getViewportsInfo();
    for (const viewportInfo of viewportsInfo) {
      const { viewportId, renderingEngineId } = viewportInfo;
      const enabledElement = getEnabledElementByIds(
        viewportId,
        renderingEngineId
      );
      if (!enabledElement) {
        continue;
      }
      const viewport = enabledElement.viewport as Types.IVolumeViewport;
      const resetPan = true;
      const resetZoom = true;
      const resetToCenter = true;
      const resetRotation = true;
      const suppressEvents = true;
      if (csUtils.isGenericViewport(viewport)) {
        // Native PLANAR_NEXT has no resetCamera/resetSlabThickness; use
        // resetViewState (slice/navigation is preserved, no slab concept).
        viewport.resetViewState({
          resetPan,
          resetZoom,
          resetOrientation: resetRotation,
          resetFlip: true,
        });
      } else {
        viewport.resetCamera({
          resetPan,
          resetZoom,
          resetToCenter,
          resetRotation,
          suppressEvents,
        });
        (viewport as Types.IVolumeViewport).resetSlabThickness();
      }
      const { element } = viewport;
      let annotations = this._getAnnotations(enabledElement);
      annotations = this.filterInteractableAnnotationsForElement(
        element,
        annotations
      );
      if (annotations.length) {
        removeAnnotation(annotations[0].annotationUID);
      }
      viewport.render();
    }

    this._computeToolCenter(viewportsInfo);
  };

  computeToolCenter = () => {
    const viewportsInfo = this._getViewportsInfo();
    this._computeToolCenter(viewportsInfo);
  };

  /**
   * When activated, it initializes the crosshairs. It begins by computing
   * the intersection of viewports associated with the crosshairs instance.
   * When all three views are accessible, the intersection (e.g., crosshairs tool centre)
   * will be an exact point in space; however, with two viewports, because the
   * intersection of two planes is a line, it assumes the last view is between the centre
   * of the two rendering viewports.
   * @param viewportsInfo Array of viewportInputs which each item containing `{viewportId, renderingEngineId}`
   */
  _computeToolCenter = (viewportsInfo): void => {
    this._debugLog('_computeToolCenter:start', {
      viewportsCount: viewportsInfo?.length ?? 0,
      toolCenter: this._formatPoint3(this.toolCenter),
      lastValidToolCenter: this._formatPoint3(this._lastValidToolCenter),
      ownFrameOfReferenceUID: this._getOwnFrameOfReferenceUID(),
      viewports: this._getViewportsDebugInfo(viewportsInfo || []),
    });

    if (!viewportsInfo.length) {
      this._debugLog('_computeToolCenter:skip:no-viewports');
      return;
    }

    viewportsInfo.forEach((viewportInfo) => {
      this.initializeViewport(viewportInfo);
    });

    // Keep the per-viewport STACK_NEW_IMAGE listeners (reverse-sync for
    // scrolling — see `_handleStackNewImage`) in sync with the current
    // viewport membership. Cheap/idempotent; only binds to StackViewports.
    this._syncStackImageListeners(viewportsInfo);

    // Every viewport is a subscriber to the shared point of its
    // FrameOfReferenceUID. Adopt the manager's point if it has one, else reuse
    // the last-known-valid center, else initialize from the own focal point.
    const forUID = this._getOwnFrameOfReferenceUID();
    const manager = getWorldPointManager();
    const sharedCenter = forUID ? manager.getPoint(forUID) : null;
    if (sharedCenter) {
      this._debugLog('_computeToolCenter:using-manager-point', {
        sharedCenter: this._formatPoint3(sharedCenter),
      });
      this._commitToolCenter(sharedCenter);
      this.setToolCenter(sharedCenter, /* suppressEvents = */ true);
      triggerAnnotationRenderForViewportIds(
        viewportsInfo.map(({ viewportId }) => viewportId)
      );
      return;
    }

    // 2+ orthogonal viewports: derive the center from their camera plane
    // intersection. Returns null for a single viewport (falls through below).
    const recomputed = this._recomputeToolCenterFromAbsoluteCameras({
      emitEvent: true,
      updateViewportCameras: true,
    });

    if (!recomputed) {
      if (this._lastValidToolCenter) {
        this._commitToolCenter(this._lastValidToolCenter);
        this._syncStackSlices(viewportsInfo, this.toolCenter);
      } else {
        this._initializeToolCenterFromOwnViewport(viewportsInfo);
      }
      triggerAnnotationRenderForViewportIds(
        viewportsInfo.map(({ viewportId }) => viewportId)
      );
    }

    this._debugLog('_computeToolCenter:end', {
      toolCenter: this._formatPoint3(this.toolCenter),
      lastValidToolCenter: this._formatPoint3(this._lastValidToolCenter),
    });
  };

  /**
   * Returns the FrameOfReferenceUID shared by this toolGroup's viewports, or
   * the last-known one / undefined if none is currently determinable.
   */
  _getOwnFrameOfReferenceUID = (): string | undefined => {
    const viewportsInfo = this._getViewportsInfo();
    for (const { viewportId, renderingEngineId } of viewportsInfo) {
      const enabledElement = getEnabledElementByIds(
        viewportId,
        renderingEngineId
      );
      if (enabledElement?.FrameOfReferenceUID) {
        this._lastKnownFrameOfReferenceUID = enabledElement.FrameOfReferenceUID;
        return enabledElement.FrameOfReferenceUID;
      }
    }
    return this._lastKnownFrameOfReferenceUID ?? undefined;
  };

  /**
   * Final fallback: initializes the shared tool center from the own
   * viewport's camera focal point (defined for every viewport type).
   */
  _initializeToolCenterFromOwnViewport = (
    viewportsInfo: { viewportId: string; renderingEngineId: string }[]
  ): void => {
    for (const { viewportId, renderingEngineId } of viewportsInfo) {
      const enabledElement = getEnabledElementByIds(
        viewportId,
        renderingEngineId
      );
      if (!enabledElement) {
        continue;
      }

      const camera = getViewportICamera(enabledElement.viewport);
      const focalPoint = camera?.focalPoint as Types.Point3 | undefined;
      if (!focalPoint || !this._isFinitePoint3(focalPoint)) {
        continue;
      }

      const forUID = enabledElement.FrameOfReferenceUID;
      if (forUID) {
        const manager = getWorldPointManager();
        manager.initializePoint(forUID, focalPoint);
      }
      this._commitToolCenter(focalPoint);
      return;
    }
  };

  /**
   * Schedules a debounced _computeToolCenter call. When multiple viewports
   * are added rapidly (e.g. MPR triad: axial+sagittal+coronal), this
   * coalesces them into a single recompute after the debounce delay,
   * allowing cameras to settle before computing the intersection.
   */
  _scheduleDebouncedRecompute = (): void => {
    if (this._debouncedRecomputeHandle) {
      clearTimeout(this._debouncedRecomputeHandle);
    }

    this._debouncedRecomputeHandle = setTimeout(() => {
      this._debouncedRecomputeHandle = null;
      this._computeToolCenter(this._getViewportsInfo());
    }, this._debounceDelayMs);
  };

  _cancelDebouncedRecompute = (): void => {
    if (this._debouncedRecomputeHandle) {
      clearTimeout(this._debouncedRecomputeHandle);
      this._debouncedRecomputeHandle = null;
    }
  };

  /**
   * Auto-scrolls any StackViewport(s) to the image index closest to
   * `worldPoint`, only if it differs from the currently displayed one.
   * Volume/Volume3D viewports are left untouched (no per-slice concept).
   */
  _syncStackSlices = (
    viewportsInfo: { viewportId: string; renderingEngineId: string }[],
    worldPoint: Types.Point3,
    excludeKey?: string
  ): void => {
    // setImageIdIndex() below fires STACK_NEW_IMAGE, which our own
    // _handleStackNewImage would misread as a user scroll and re-broadcast a
    // tool center shift (feedback loop). Guard against that.
    const previousIgnoreFiredEvents = this._ignoreFiredEvents;
    this._ignoreFiredEvents = true;
    try {
      viewportsInfo.forEach(({ viewportId, renderingEngineId }) => {
        const key = this._toViewportKey(renderingEngineId, viewportId);
        // Never auto-scroll the viewport being dragged (or otherwise
        // excluded): it must stay on the slice the user is looking at.
        if (excludeKey && key === excludeKey) {
          return;
        }

        const enabledElement = getEnabledElementByIds(
          viewportId,
          renderingEngineId
        );
        if (!enabledElement) {
          return;
        }

        const { viewport } = enabledElement;
        if (!(viewport instanceof StackViewport)) {
          return;
        }

        const closestIndex = csUtils.getClosestStackImageIndexForPoint(
          worldPoint,
          viewport
        );

        if (
          closestIndex === null ||
          closestIndex === undefined ||
          closestIndex === viewport.getCurrentImageIdIndex()
        ) {
          return;
        }

        // setImageIdIndex() fires STACK_NEW_IMAGE asynchronously, after the
        // _ignoreFiredEvents guard is reset. Record the requested index so
        // _handleStackNewImage can recognize and ignore exactly that transition.
        if (this._stackImageListeners.has(key)) {
          this._pendingProgrammaticSliceIndex.set(key, closestIndex);
        }

        this._debugLog('_syncStackSlices:setImageIdIndex', {
          viewportId,
          from: viewport.getCurrentImageIdIndex(),
          to: closestIndex,
          worldPoint: this._formatPoint3(worldPoint),
        });

        viewport.setImageIdIndex(closestIndex);
        viewport.render();
      });
    } finally {
      this._ignoreFiredEvents = previousIgnoreFiredEvents;
    }
  };

  setToolCenter(toolCenter: Types.Point3, suppressEvents = false): void {
    const viewportsInfo = this._getViewportsInfo();
    const previousIgnoreFiredEvents = this._ignoreFiredEvents;
    this._ignoreFiredEvents = true;
    try {
      viewportsInfo.forEach(({ renderingEngineId, viewportId }) => {
        const renderingEngine = getRenderingEngine(renderingEngineId);
        if (!renderingEngine) {
          return;
        }

        const viewport = renderingEngine.getViewport(viewportId);
        if (!viewport) {
          return;
        }

        const camera = getViewportICamera(viewport);
        const { focalPoint, position, viewPlaneNormal } = camera;

        const delta = [
          toolCenter[0] - focalPoint[0],
          toolCenter[1] - focalPoint[1],
          toolCenter[2] - focalPoint[2],
        ];

        // Project the delta onto the viewport's own view plane normal so we
        // only "scroll" the reformat plane to the slice containing the tool
        // center (as MPR does), never panning it in-plane. StackViewports snap
        // to the closest slice separately via `_syncStackSlices`.
        const scrollDelta = (() => {
          const scroll =
            delta[0] * viewPlaneNormal[0] +
            delta[1] * viewPlaneNormal[1] +
            delta[2] * viewPlaneNormal[2];

          return [
            scroll * viewPlaneNormal[0],
            scroll * viewPlaneNormal[1],
            scroll * viewPlaneNormal[2],
          ];
        })();

        const newFocalPoint: Types.Point3 = [
          focalPoint[0] + scrollDelta[0],
          focalPoint[1] + scrollDelta[1],
          focalPoint[2] + scrollDelta[2],
        ];
        const newPosition: Types.Point3 = [
          position[0] + scrollDelta[0],
          position[1] + scrollDelta[1],
          position[2] + scrollDelta[2],
        ];

        if (csUtils.isGenericViewport(viewport)) {
          jumpToFocalPoint(viewport, newFocalPoint);
        } else {
          viewport.setCamera({
            focalPoint: newFocalPoint,
            position: newPosition,
          });
        }

        viewport.render();
      });
    } finally {
      this._ignoreFiredEvents = previousIgnoreFiredEvents;
    }

    this._commitToolCenter(toolCenter);

    if (!suppressEvents) {
      const forUID = this._getOwnFrameOfReferenceUID();
      const manager = getWorldPointManager();
      const currentPoint = forUID ? manager.getPoint(forUID) : null;
      triggerEvent(eventTarget, Events.CROSSHAIR_TOOL_CENTER_CHANGED, {
        toolGroupId: this.toolGroupId,
        toolCenter: currentPoint ?? this.toolCenter,
        FrameOfReferenceUID: forUID,
      });
    }
  }

  /**
   * addNewAnnotation acts as jump for the crosshairs tool. It is called when
   * the user clicks on the image. It does not store the annotation in the stateManager though.
   *
   * @param evt - The mouse event
   * @param interactionType - The type of interaction (e.g., mouse, touch, etc.)
   * @returns Crosshairs annotation
   */
  addNewAnnotation = (
    evt: EventTypes.InteractionEventType
  ): CrosshairsAnnotation => {
    const eventDetail = evt.detail;
    const { element } = eventDetail;

    const enabledElementForModality = getEnabledElement(element);
    if (enabledElementForModality?.viewport) {
      const modality = this._getViewportModality(enabledElementForModality.viewport);
      if (modality && !getWorldPointManager().isModalitySupported(modality)) {
        evt.preventDefault();
        return null;
      }
    }

    const { currentPoints } = eventDetail;
    const jumpWorld = currentPoints.world;

    this._syncVolumeListenersWithToolGroup();
    this._recomputeToolCenterFromAbsoluteCameras({
      emitEvent: false,
      updateViewportCameras: false,
    });

    const enabledElement = getEnabledElement(element);
    const { viewport } = enabledElement;
    this._jump(enabledElement, jumpWorld);

    const annotations = this._getAnnotations(enabledElement);
    const filteredAnnotations = this.filterInteractableAnnotationsForElement(
      viewport.element,
      annotations
    );

    // viewport Annotation
    const { data } = filteredAnnotations[0];

    const { rotationPoints } = data.handles;
    const viewportIdArray = [];
    // put all the draggable reference lines in the viewportIdArray
    for (let i = 0; i < rotationPoints.length - 1; ++i) {
      const otherViewport = rotationPoints[i][1];
      const viewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      const viewportDraggableRotatable =
        this._getReferenceLineDraggableRotatable(otherViewport.id);
      if (!viewportControllable || !viewportDraggableRotatable) {
        continue;
      }
      viewportIdArray.push(otherViewport.id);
      // rotation handles are two per viewport
      i++;
    }

    data.activeViewportIds = [...viewportIdArray];
    // set translation operation
    data.handles.activeOperation = OPERATION.DRAG;

    evt.preventDefault();

    hideElementCursor(element);

    this._activateModify(element);
    return filteredAnnotations[0];
  };

  cancel = () => {
    console.log('Not implemented yet');
  };

  /**
   * It checks if the mouse click is near crosshairs handles, if yes
   * it returns the handle location. If the mouse click is not near any
   * of the handles, it does not return anything.
   *
   * @param element - The element that the tool is attached to.
   * @param annotation - The annotation object associated with the annotation
   * @param canvasCoords - The coordinates of the mouse click on canvas
   * @param proximity - The distance from the mouse cursor to the point
   * that is considered "near".
   * @returns The handle that is closest to the cursor, or null if the cursor
   * is not near any of the handles.
   */
  getHandleNearImagePoint(
    element: HTMLDivElement,
    annotation: Annotation,
    canvasCoords: Types.Point2,
    proximity: number
  ): ToolHandle | undefined {
    const enabledElement = getEnabledElement(element);
    const { viewport } = enabledElement;

    let point = this._getRotationHandleNearImagePoint(
      viewport,
      annotation,
      canvasCoords,
      proximity
    );

    if (point !== null) {
      return point;
    }

    point = this._getSlabThicknessHandleNearImagePoint(
      viewport,
      annotation,
      canvasCoords,
      proximity
    );

    if (point !== null) {
      return point;
    }
  }

  handleSelectedCallback = (
    evt: EventTypes.InteractionEventType,
    annotation: Annotation
  ): void => {
    const eventDetail = evt.detail;
    const { element } = eventDetail;
    annotation.highlighted = true;

    // NOTE: handle index/coordinates aren't used when dragging — translation,
    // rotation and slab thickness operate on the camera variables, not handles
    // (handle points are derived in renderTool from the cameras).
    this._activateModify(element);

    hideElementCursor(element);

    evt.preventDefault();
  };

  /**
   * It returns if the canvas point is near the provided crosshairs annotation in the
   * provided element or not. A proximity is passed to the function to determine the
   * proximity of the point to the annotation in number of pixels.
   *
   * @param element - HTML Element
   * @param annotation - Annotation
   * @param canvasCoords - Canvas coordinates
   * @param proximity - Proximity to tool to consider
   * @returns Boolean, whether the canvas point is near tool
   */
  isPointNearTool = (
    element: HTMLDivElement,
    annotation: CrosshairsAnnotation,
    canvasCoords: Types.Point2,
    proximity: number
  ): boolean => {
    const enabledElement = getEnabledElement(element);
    if (enabledElement?.viewport) {
      const modality = this._getViewportModality(enabledElement.viewport);
      if (modality && !getWorldPointManager().isModalitySupported(modality)) {
        return false;
      }
    }

    if (this._pointNearTool(element, annotation, canvasCoords, 6)) {
      return true;
    }

    return false;
  };

  toolSelectedCallback = (
    evt: EventTypes.InteractionEventType,
    annotation: Annotation,
    interactionType: InteractionTypes
  ): void => {
    const eventDetail = evt.detail;
    const { element } = eventDetail;
    annotation.highlighted = true;
    this._activateModify(element);

    hideElementCursor(element);

    evt.preventDefault();
  };

  onCameraModified = (evt) => {
    const eventDetail = evt.detail;
    const { element } = eventDetail;
    const enabledElement = getEnabledElement(element);
    if (!enabledElement) {
      return;
    }

    const { renderingEngine } = enabledElement;
    const viewport = enabledElement.viewport as Types.IVolumeViewport;
    this._syncVolumeListenersWithToolGroup();

    if (this._ignoreFiredEvents) {
      return;
    }

    const isSourceInToolGroup = this._getViewportsInfo().some(
      ({ viewportId, renderingEngineId }) =>
        viewportId === viewport.id && renderingEngineId === renderingEngine.id
    );

    if (!isSourceInToolGroup) {
      return;
    }

    const annotations = this._getAnnotations(enabledElement);
    const filteredToolAnnotations =
      this.filterInteractableAnnotationsForElement(element, annotations);

    // viewport that the camera modified is originating from
    const viewportAnnotation =
      filteredToolAnnotations[0] as CrosshairsAnnotation;

    const currentCamera = getViewportICamera(viewport);
    if (viewportAnnotation) {
      viewportAnnotation.metadata.cameraPosition = [...currentCamera.position];
      viewportAnnotation.metadata.cameraFocalPoint = [
        ...currentCamera.focalPoint,
      ];
    }

    // Only MPR reformat VolumeViewports update the shared point from a camera
    // delta. StackViewport slice changes are handled via STACK_NEW_IMAGE
    // (`_handleStackNewImage`), and VolumeViewport3D free-orbit moves must
    // never alter the point (they only re-project it on the next render).
    if (
      !(viewport instanceof StackViewport) &&
      !(viewport instanceof VolumeViewport3D)
    ) {
      this._updateToolCenterFromCameraDelta(viewport, evt);
    }

    // AutoPan modification
    if (this.configuration.autoPan?.enabled) {
      const toolGroup = getToolGroupForViewport(
        viewport.id,
        renderingEngine.id
      );

      const otherViewportIds = toolGroup
        .getViewportIds()
        .filter((id) => id !== viewport.id);

      otherViewportIds.forEach((viewportId) => {
        this._autoPanViewportIfNecessary(viewportId, renderingEngine);
      });
    }

    const requireSameOrientation = false;
    const viewportIdsToRender = getViewportIdsWithToolToRender(
      element,
      this.getToolName(),
      requireSameOrientation
    );

    triggerAnnotationRenderForViewportIds(viewportIdsToRender);
  };

  /**
   * Keeps `_stackImageListeners` in sync with the toolGroup's current
   * StackViewport membership. Safe/cheap to call repeatedly.
   */
  _syncStackImageListeners = (viewportsInfo): void => {
    const currentKeys = new Set<string>();

    viewportsInfo.forEach(({ viewportId, renderingEngineId }) => {
      const key = this._toViewportKey(renderingEngineId, viewportId);
      currentKeys.add(key);

      if (this._stackImageListeners.has(key)) {
        return;
      }

      const enabledElement = getEnabledElementByIds(
        viewportId,
        renderingEngineId
      );
      if (!enabledElement) {
        return;
      }

      const { viewport } = enabledElement;
      if (!(viewport instanceof StackViewport)) {
        return;
      }

      const { element } = viewport;
      const listener = ((evt: CustomEvent) => {
        this._handleStackNewImage(
          key,
          viewportId,
          viewport,
          evt.detail
        );
      }) as EventListener;

      element.addEventListener(Enums.Events.STACK_NEW_IMAGE, listener);
      this._stackImageListeners.set(key, { element, listener });

      // Mark as "seen" only if an image is already loaded; otherwise leave it
      // unseen so _handleStackNewImage's "first image seen" branch handles it
      // once the image finishes loading.
      if (viewport.getImageData?.()) {
        this._seenStackViewports.add(key);
      }
    });

    for (const [key, { element, listener }] of this
      ._stackImageListeners) {
      if (!currentKeys.has(key)) {
        element.removeEventListener(Enums.Events.STACK_NEW_IMAGE, listener);
        this._stackImageListeners.delete(key);
        this._seenStackViewports.delete(key);
      }
    }
  };

  _unbindStackImageListeners = (): void => {
    for (const {
      element,
      listener,
    } of this._stackImageListeners.values()) {
      element.removeEventListener(Enums.Events.STACK_NEW_IMAGE, listener);
    }
    this._stackImageListeners.clear();
    this._seenStackViewports.clear();
    this._pendingProgrammaticSliceIndex.clear();
  };

  /**
   * Reverse sync for StackViewports: when the user scrolls a stack's series,
   * shift the shared tool center along that viewport's view plane normal and
   * broadcast it outward. Listens to STACK_NEW_IMAGE (not CAMERA_MODIFIED)
   * since stack scrolling doesn't reliably fire CAMERA_MODIFIED.
   */
  _handleStackNewImage = (
    key: string,
    viewportId: string,
    viewport,
    eventDetail?: { imageIdIndex?: number }
  ): void => {
    // Guard against our own programmatic setImageIdIndex() calls if they
    // resolve synchronously (the async case is handled just below).
    if (this._ignoreFiredEvents) {
      return;
    }

    // The programmatic setImageIdIndex() (see _syncStackSlices) fires
    // STACK_NEW_IMAGE asynchronously, after _ignoreFiredEvents was reset.
    // Recognize and ignore exactly the transition we requested.
    const expectedIndex = this._pendingProgrammaticSliceIndex.get(key);
    if (expectedIndex !== undefined) {
      this._pendingProgrammaticSliceIndex.delete(key);
      if (eventDetail?.imageIdIndex === expectedIndex) {
        // Our own programmatic slice snap; must not overwrite the manager point.
        return;
      }
      // Otherwise the user scrolled again before it finished loading — treat
      // this image as genuine.
    }

    const camera = getViewportICamera(viewport);
    const newFocalPoint = camera?.focalPoint as Types.Point3 | undefined;
    if (!newFocalPoint || !this._isFinitePoint3(newFocalPoint)) {
      return;
    }

    this._debugLog('_handleStackNewImage:user-scroll', {
      viewportId,
      imageIdIndex: eventDetail?.imageIdIndex,
      newFocalPoint: this._formatPoint3(newFocalPoint),
      toolCenter: this._formatPoint3(this.toolCenter),
    });

    // First image ever seen for this viewport — nothing to compare against
    // yet, and its camera/FrameOfReferenceUID have just settled to real
    // values. Re-run initializeViewport to recreate the annotation with a
    // fresh FrameOfReferenceUID, then request a render.
    const isFirstImageSeen = !this._seenStackViewports.has(key);
    this._seenStackViewports.add(key);
    if (isFirstImageSeen) {
      const renderingEngineId = this._getViewportsInfo().find(
        (info) => info.viewportId === viewportId
      )?.renderingEngineId;
      if (renderingEngineId) {
        this.initializeViewport({ viewportId, renderingEngineId });
      }

      const forUID = this._getOwnFrameOfReferenceUID();
      const manager = getWorldPointManager();
      if (
        (forUID && !manager.isInitialized(forUID)) ||
        !this._isFinitePoint3(this.toolCenter) ||
        this._isNearZeroPoint3(this.toolCenter)
      ) {
        if (forUID) {
          manager.initializePoint(forUID, newFocalPoint);
        }
        this._commitToolCenter(newFocalPoint);
      }

      triggerAnnotationRenderForViewportIds([viewportId]);
      return;
    }

    // Self-correct rather than diffing against a stale camera snapshot:
    // measure how far the shared tool center currently sits from this
    // viewport's slice plane (along its normal) and shift by exactly that.
    // Idempotent and always relative to the latest state.
    const { viewPlaneNormal } = camera;
    const toCenterDelta: Types.Point3 = [
      newFocalPoint[0] - this.toolCenter[0],
      newFocalPoint[1] - this.toolCenter[1],
      newFocalPoint[2] - this.toolCenter[2],
    ];

    const distanceAlongNormal =
      toCenterDelta[0] * viewPlaneNormal[0] +
      toCenterDelta[1] * viewPlaneNormal[1] +
      toCenterDelta[2] * viewPlaneNormal[2];

    // Already on (or very close to) the slice the shared tool center
    // already points to — nothing to do (also naturally covers the case
    // where this event is a redundant re-fire).
    if (Math.abs(distanceAlongNormal) < 1e-2) {
      return;
    }

    const shift: Types.Point3 = [
      distanceAlongNormal * viewPlaneNormal[0],
      distanceAlongNormal * viewPlaneNormal[1],
      distanceAlongNormal * viewPlaneNormal[2],
    ];

    const newToolCenter: Types.Point3 = [
      this.toolCenter[0] + shift[0],
      this.toolCenter[1] + shift[1],
      this.toolCenter[2] + shift[2],
    ];

    const forUID = this._getOwnFrameOfReferenceUID();
    const manager = getWorldPointManager();
    const renderingEngineId = this._getViewportsInfo().find(
      (info) => info.viewportId === viewportId
    )?.renderingEngineId;

    // Update the shared point in the manager (source of truth); it notifies
    // every other subscriber. We deliberately don't call _syncStackSlices for
    // this viewport — it's already on the correct slice.
    this._commitToolCenter(newToolCenter, {
      senderViewportId: viewportId,
      senderRenderingEngineId: renderingEngineId,
    });

    triggerEvent(eventTarget, Events.CROSSHAIR_TOOL_CENTER_CHANGED, {
      toolGroupId: this.toolGroupId,
      toolCenter: forUID ? (manager.getPoint(forUID) ?? newToolCenter) : newToolCenter,
      FrameOfReferenceUID: forUID,
    });

    triggerAnnotationRenderForViewportIds(
      this._getViewportsInfo().map(({ viewportId }) => viewportId)
    );
  };

  onResetCamera = (evt) => {
    this.resetCrosshairs();
  };

  mouseMoveCallback = (
    evt: EventTypes.MouseMoveEventType,
    filteredToolAnnotations: Annotations
  ): boolean => {
    const { element, currentPoints } = evt.detail;
    const canvasCoords = currentPoints.canvas;
    let imageNeedsUpdate = false;

    const enabledElement = getEnabledElement(element);
    if (enabledElement?.viewport) {
      const modality = this._getViewportModality(enabledElement.viewport);
      if (modality && !getWorldPointManager().isModalitySupported(modality)) {
        return imageNeedsUpdate;
      }
    }

    // `filteredToolAnnotations` can be undefined when there is no annotation
    // for this element yet (e.g. before initializeViewport has run).
    if (!filteredToolAnnotations?.length) {
      return imageNeedsUpdate;
    }

    for (let i = 0; i < filteredToolAnnotations.length; i++) {
      const annotation = filteredToolAnnotations[i] as CrosshairsAnnotation;

      if (isAnnotationLocked(annotation.annotationUID)) {
        continue;
      }

      const { data, highlighted } = annotation;
      if (!data.handles) {
        continue;
      }

      const previousActiveOperation = data.handles.activeOperation;
      const previousActiveViewportIds =
        data.activeViewportIds && data.activeViewportIds.length > 0
          ? [...data.activeViewportIds]
          : [];

      // This init are necessary, because when we move the mouse they are not cleaned by _endCallback
      data.activeViewportIds = [];
      data.handles.activeOperation = null;

      const handleNearImagePoint = this.getHandleNearImagePoint(
        element,
        annotation,
        canvasCoords,
        6
      );

      let near = false;
      if (handleNearImagePoint) {
        near = true;
      } else {
        near = this._pointNearTool(element, annotation, canvasCoords, 6);
      }

      const nearToolAndNotMarkedActive = near && !highlighted;
      const notNearToolAndMarkedActive = !near && highlighted;
      if (nearToolAndNotMarkedActive || notNearToolAndMarkedActive) {
        annotation.highlighted = !highlighted;
        imageNeedsUpdate = true;
      } else if (
        data.handles.activeOperation !== previousActiveOperation ||
        !this._areViewportIdArraysEqual(
          data.activeViewportIds,
          previousActiveViewportIds
        )
      ) {
        imageNeedsUpdate = true;
      }
    }

    return imageNeedsUpdate;
  };

  filterInteractableAnnotationsForElement = (element, annotations) => {
    if (!annotations || !annotations.length) {
      return [];
    }

    const enabledElement = getEnabledElement(element);
    const { viewportId } = enabledElement;

    const viewportUIDSpecificCrosshairs = annotations.filter(
      (annotation) => annotation.data.viewportId === viewportId
    );

    return viewportUIDSpecificCrosshairs;
  };

  /**
   * renders the crosshairs lines and handles in the requestAnimationFrame callback
   *
   * @param enabledElement - The Cornerstone's enabledElement.
   * @param svgDrawingHelper - The svgDrawingHelper providing the context for drawing.
   */
  renderAnnotation = (
    enabledElement: Types.IEnabledElement,
    svgDrawingHelper: SVGDrawingHelper
  ): boolean => {
    let renderStatus = false;
    const { viewport, renderingEngine } = enabledElement;
    const { element } = viewport;
    const annotations = this._getAnnotations(enabledElement);
    const camera = getViewportICamera(viewport);
    const filteredToolAnnotations =
      this.filterInteractableAnnotationsForElement(element, annotations);

    // viewport Annotation
    const viewportAnnotation = filteredToolAnnotations[0];
    if (!annotations?.length || !viewportAnnotation?.data) {
      // No annotations yet, and didn't just create it as we likely don't have a FrameOfReference/any data loaded yet.
      return renderStatus;
    }

    const annotationUID = viewportAnnotation.annotationUID;

    const viewportsInfo = this._getViewportsInfo();

    const viewportModality = this._getViewportModality(viewport);
    if (viewportModality) {
      const manager = getWorldPointManager();
      if (!manager.isModalitySupported(viewportModality)) {
        return renderStatus;
      }
    }

    // A lone StackViewport (single subscriber for its FOR) has no other
    // orientation planes to draw reference lines against — but we still fall
    // through to render the center handle below.

    // Get cameras/canvases for each of these.
    // -- Get two world positions for this canvas in this line (e.g. the diagonal)
    // -- Convert these world positions to this canvas.
    // -- Extend/confine this line to fit in this canvas.
    // -- Render this line.
    const { clientWidth, clientHeight } = getDisplayedCanvasSize(viewport);
    const canvasDiagonalLength = Math.sqrt(
      clientWidth * clientWidth + clientHeight * clientHeight
    );
    const canvasMinDimensionLength = Math.min(clientWidth, clientHeight);

    const data = viewportAnnotation.data;
    const crosshairCenterCanvas = viewport.worldToCanvas(this.toolCenter);

    // Reference lines / rotation / slab-thickness handles are only meaningful
    // between viewports with distinct orientation planes. This filter returns
    // an empty array for a single-viewport FOR, in which case only the
    // `centerHandle` circle (rendered further below) is drawn.
    const otherViewportAnnotations =
      this._filterAnnotationsByUniqueViewportOrientations(
        enabledElement,
        annotations
      );

    const referenceLines = [];

    // get canvas information for points and lines (canvas box, canvas horizontal distances)
    const canvasBox = [0, 0, clientWidth, clientHeight];

    otherViewportAnnotations.forEach((annotation) => {
      const { data } = annotation;

      data.handles.toolCenter = this.toolCenter;

      const otherViewport = renderingEngine.getViewport(
        data.viewportId
      ) as Types.IVolumeViewport;

      // Reference lines / slab-thickness handles are only meaningful for
      // volume viewports with orientation planes. Skip StackViewports (which
      // have no slab API) to avoid calling volume-only methods on them.
      if (
        !otherViewport ||
        otherViewport instanceof StackViewport ||
        typeof otherViewport.getSlabThickness !== 'function'
      ) {
        return;
      }

      const otherCamera = getViewportICamera(otherViewport);

      const otherViewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      const otherViewportDraggableRotatable =
        this._getReferenceLineDraggableRotatable(otherViewport.id);
      const otherViewportSlabThicknessControlsOn =
        this._getReferenceLineSlabThicknessControlsOn(otherViewport.id);

      // get coordinates for the reference line
      const { clientWidth, clientHeight } =
        getDisplayedCanvasSize(otherViewport);
      const otherCanvasDiagonalLength = Math.sqrt(
        clientWidth * clientWidth + clientHeight * clientHeight
      );
      const otherCanvasCenter: Types.Point2 = [
        clientWidth * 0.5,
        clientHeight * 0.5,
      ];
      const otherViewportCenterWorld =
        otherViewport.canvasToWorld(otherCanvasCenter);

      const direction: Types.Point3 = [0, 0, 0];
      vtkMath.cross(
        camera.viewPlaneNormal,
        otherCamera.viewPlaneNormal,
        direction
      );
      vtkMath.normalize(direction);
      vtkMath.multiplyScalar(
        <Types.Point3>direction,
        otherCanvasDiagonalLength
      );

      const pointWorld0: Types.Point3 = [0, 0, 0];
      vtkMath.add(otherViewportCenterWorld, direction, pointWorld0);

      const pointWorld1: Types.Point3 = [0, 0, 0];
      vtkMath.subtract(otherViewportCenterWorld, direction, pointWorld1);

      const pointCanvas0 = viewport.worldToCanvas(pointWorld0);

      const otherViewportCenterCanvas = viewport.worldToCanvas(
        otherViewportCenterWorld
      );

      const canvasUnitVectorFromCenter = vec2.create();
      vec2.subtract(
        canvasUnitVectorFromCenter,
        pointCanvas0,
        otherViewportCenterCanvas
      );
      vec2.normalize(canvasUnitVectorFromCenter, canvasUnitVectorFromCenter);

      // Graphic:
      // Mid -> SlabThickness handle
      // Short -> Rotation handle
      //                           Long
      //                            |
      //                            |
      //                            |
      //                           Mid
      //                            |
      //                            |
      //                            |
      //                          Short
      //                            |
      //                            |
      //                            |
      // Long --- Mid--- Short--- Center --- Short --- Mid --- Long
      //                            |
      //                            |
      //                            |
      //                          Short
      //                            |
      //                            |
      //                            |
      //                           Mid
      //                            |
      //                            |
      //                            |
      //                           Long
      const canvasVectorFromCenterLong = vec2.create();

      vec2.scale(
        canvasVectorFromCenterLong,
        canvasUnitVectorFromCenter,
        canvasDiagonalLength * 100
      );
      const canvasVectorFromCenterShort = vec2.create();
      vec2.scale(
        canvasVectorFromCenterShort,
        canvasUnitVectorFromCenter,
        // Chosen 0.2 because is half of 0.4.
        canvasMinDimensionLength * 0.2
      );
      const canvasVectorFromCenterStart = vec2.create();
      // Calculate center gap using ratio if provided, else fallback to pixel value
      const mobileConfig = this.configuration.mobile;
      const activeConfiguration = mobileConfig?.enabled
        ? mobileConfig
        : this.configuration;
      const { referenceLinesCenterGapRatio } = activeConfiguration;
      const minimalCrosshairConfig = getMinimalCrosshairConfig(
        this.configuration
      );

      const centerGap =
        referenceLinesCenterGapRatio > 0
          ? canvasMinDimensionLength * referenceLinesCenterGapRatio
          : this.configuration.referenceLinesCenterGapRadius;

      vec2.scale(
        canvasVectorFromCenterStart,
        canvasUnitVectorFromCenter,
        // Don't put a gap if the the third view is missing
        otherViewportAnnotations.length === 2 ? centerGap : 0
      );

      // Computing Reference start and end (4 lines per viewport in case of 3 view MPR)
      const refLinePointOne = vec2.create();
      const refLinePointTwo = vec2.create();
      const refLinePointThree = vec2.create();
      const refLinePointFour = vec2.create();

      let refLinesCenter = vec2.clone(crosshairCenterCanvas);
      if (!otherViewportDraggableRotatable || !otherViewportControllable) {
        refLinesCenter = vec2.clone(otherViewportCenterCanvas);
      }

      vec2.add(refLinePointOne, refLinesCenter, canvasVectorFromCenterStart);
      vec2.add(refLinePointTwo, refLinesCenter, canvasVectorFromCenterLong);
      vec2.subtract(
        refLinePointThree,
        refLinesCenter,
        canvasVectorFromCenterStart
      );
      vec2.subtract(
        refLinePointFour,
        refLinesCenter,
        canvasVectorFromCenterLong
      );

      // Clipping lines to be only included in a box (canvas), we don't want
      // the lines goes beyond canvas
      liangBarksyClip(refLinePointOne, refLinePointTwo, canvasBox);
      liangBarksyClip(refLinePointThree, refLinePointFour, canvasBox);

      if (minimalCrosshairConfig.enabled) {
        const minimalCanvasVectorFromCenterLong = vec2.create();
        vec2.scale(
          minimalCanvasVectorFromCenterLong,
          canvasUnitVectorFromCenter,
          minimalCrosshairConfig.lineLengthInPx
        );

        vec2.add(refLinePointOne, refLinesCenter, canvasVectorFromCenterStart);
        vec2.add(
          refLinePointTwo,
          refLinePointOne,
          minimalCanvasVectorFromCenterLong
        );
        vec2.subtract(
          refLinePointThree,
          refLinesCenter,
          canvasVectorFromCenterStart
        );
        vec2.subtract(
          refLinePointFour,
          refLinePointThree,
          minimalCanvasVectorFromCenterLong
        );

        liangBarksyClip(refLinePointOne, refLinePointTwo, canvasBox);
        liangBarksyClip(refLinePointThree, refLinePointFour, canvasBox);
      }

      // Computing rotation handle positions
      const rotHandleOne = getSegmentMidpoint(refLinePointOne, refLinePointTwo);
      const rotHandleTwo = getSegmentMidpoint(
        refLinePointThree,
        refLinePointFour
      );

      // Computing SlabThickness (st below) position

      // SlabThickness center in canvas
      let stHandlesCenterCanvas = vec2.clone(crosshairCenterCanvas);
      if (
        !otherViewportDraggableRotatable &&
        otherViewportSlabThicknessControlsOn
      ) {
        stHandlesCenterCanvas = vec2.clone(otherViewportCenterCanvas);
      }

      // SlabThickness center in world
      let stHandlesCenterWorld: Types.Point3 = [...this.toolCenter];
      if (
        !otherViewportDraggableRotatable &&
        otherViewportSlabThicknessControlsOn
      ) {
        stHandlesCenterWorld = [...otherViewportCenterWorld];
      }

      const worldUnitVectorFromCenter: Types.Point3 = [0, 0, 0];
      vtkMath.subtract(pointWorld0, pointWorld1, worldUnitVectorFromCenter);
      vtkMath.normalize(worldUnitVectorFromCenter);

      const { viewPlaneNormal } = camera;
      // @ts-ignore // Todo: fix after vtk pr merged
      const { matrix } = vtkMatrixBuilder
        .buildFromDegree()
        // @ts-ignore fix after vtk pr merged
        .rotate(90, viewPlaneNormal);

      const worldUnitOrthoVectorFromCenter: Types.Point3 = [0, 0, 0];
      vec3.transformMat4(
        worldUnitOrthoVectorFromCenter,
        worldUnitVectorFromCenter,
        matrix
      );

      const slabThicknessValue = getSlabThicknessOrDefault(otherViewport);
      const worldOrthoVectorFromCenter: Types.Point3 = [
        ...worldUnitOrthoVectorFromCenter,
      ];
      vtkMath.multiplyScalar(worldOrthoVectorFromCenter, slabThicknessValue);

      const worldVerticalRefPoint: Types.Point3 = [0, 0, 0];
      vtkMath.add(
        stHandlesCenterWorld,
        worldOrthoVectorFromCenter,
        worldVerticalRefPoint
      );

      // convert vertical world distances in canvas coordinates
      const canvasVerticalRefPoint = viewport.worldToCanvas(
        worldVerticalRefPoint
      );

      // points for slab thickness lines
      const canvasOrthoVectorFromCenter = vec2.create();
      vec2.subtract(
        canvasOrthoVectorFromCenter,
        stHandlesCenterCanvas,
        canvasVerticalRefPoint
      );

      const stLinePointOne = vec2.create();
      vec2.subtract(
        stLinePointOne,
        stHandlesCenterCanvas,
        canvasVectorFromCenterLong
      );
      vec2.add(stLinePointOne, stLinePointOne, canvasOrthoVectorFromCenter);

      const stLinePointTwo = vec2.create();
      vec2.add(
        stLinePointTwo,
        stHandlesCenterCanvas,
        canvasVectorFromCenterLong
      );
      vec2.add(stLinePointTwo, stLinePointTwo, canvasOrthoVectorFromCenter);

      liangBarksyClip(stLinePointOne, stLinePointTwo, canvasBox);

      const stLinePointThree = vec2.create();
      vec2.add(
        stLinePointThree,
        stHandlesCenterCanvas,
        canvasVectorFromCenterLong
      );
      vec2.subtract(
        stLinePointThree,
        stLinePointThree,
        canvasOrthoVectorFromCenter
      );

      const stLinePointFour = vec2.create();
      vec2.subtract(
        stLinePointFour,
        stHandlesCenterCanvas,
        canvasVectorFromCenterLong
      );
      vec2.subtract(
        stLinePointFour,
        stLinePointFour,
        canvasOrthoVectorFromCenter
      );

      liangBarksyClip(stLinePointThree, stLinePointFour, canvasBox);

      // points for slab thickness handles
      const stHandleOne = vec2.create();
      const stHandleTwo = vec2.create();
      const stHandleThree = vec2.create();
      const stHandleFour = vec2.create();

      vec2.subtract(
        stHandleOne,
        stHandlesCenterCanvas,
        canvasVectorFromCenterShort
      );
      vec2.add(stHandleOne, stHandleOne, canvasOrthoVectorFromCenter);
      vec2.add(stHandleTwo, stHandlesCenterCanvas, canvasVectorFromCenterShort);
      vec2.add(stHandleTwo, stHandleTwo, canvasOrthoVectorFromCenter);
      vec2.subtract(
        stHandleThree,
        stHandlesCenterCanvas,
        canvasVectorFromCenterShort
      );
      vec2.subtract(stHandleThree, stHandleThree, canvasOrthoVectorFromCenter);
      vec2.add(
        stHandleFour,
        stHandlesCenterCanvas,
        canvasVectorFromCenterShort
      );
      vec2.subtract(stHandleFour, stHandleFour, canvasOrthoVectorFromCenter);

      referenceLines.push([
        otherViewport,
        refLinePointOne,
        refLinePointTwo,
        refLinePointThree,
        refLinePointFour,
        stLinePointOne,
        stLinePointTwo,
        stLinePointThree,
        stLinePointFour,
        rotHandleOne,
        rotHandleTwo,
        stHandleOne,
        stHandleTwo,
        stHandleThree,
        stHandleFour,
      ]);
    });

    const newRtpoints = [];
    const newStpoints = [];
    const viewportColor = this._getReferenceLineColor(viewport.id);
    const color =
      viewportColor !== undefined ? viewportColor : 'rgb(200, 200, 200)';

    referenceLines.forEach((line, lineIndex) => {
      // get color for the reference line
      const otherViewport = line[0];
      const minimalCrosshairConfig = getMinimalCrosshairConfig(
        this.configuration
      );
      const viewportColor = this._getReferenceLineColor(otherViewport.id);
      const viewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      const viewportDraggableRotatable =
        !minimalCrosshairConfig.enabled &&
        (this._getReferenceLineDraggableRotatable(otherViewport.id) ||
          this.configuration.mobile?.enabled);
      const viewportSlabThicknessControlsOn =
        !minimalCrosshairConfig.enabled &&
        (this._getReferenceLineSlabThicknessControlsOn(otherViewport.id) ||
          this.configuration.mobile?.enabled);
      const selectedViewportId = data.activeViewportIds.find(
        (id) => id === otherViewport.id
      );

      let color =
        viewportColor !== undefined ? viewportColor : 'rgb(200, 200, 200)';

      let lineWidth = 1;

      const lineActive =
        data.handles.activeOperation !== null &&
        data.handles.activeOperation === OPERATION.DRAG &&
        selectedViewportId;

      if (lineActive) {
        lineWidth = 2.5;
      }

      let lineUID = `${lineIndex}`;
      if (
        viewportControllable &&
        (viewportDraggableRotatable || minimalCrosshairConfig.enabled)
      ) {
        lineUID = `${lineIndex}One`;
        drawLineSvg(
          svgDrawingHelper,
          annotationUID,
          lineUID,
          line[1],
          line[2],
          {
            color,
            lineWidth,
          }
        );

        lineUID = `${lineIndex}Two`;
        drawLineSvg(
          svgDrawingHelper,
          annotationUID,
          lineUID,
          line[3],
          line[4],
          {
            color,
            lineWidth,
          }
        );
      } else {
        drawLineSvg(
          svgDrawingHelper,
          annotationUID,
          lineUID,
          line[2],
          line[4],
          {
            color,
            lineWidth,
          }
        );
      }

      if (viewportControllable) {
        color =
          viewportColor !== undefined ? viewportColor : 'rgb(200, 200, 200)';

        const rotHandlesActive =
          data.handles.activeOperation === OPERATION.ROTATE;
        const rotationHandles = [line[9], line[10]];

        const rotHandleWorldOne = [
          viewport.canvasToWorld(line[9]),
          otherViewport,
          line[1],
          line[2],
        ];
        const rotHandleWorldTwo = [
          viewport.canvasToWorld(line[10]),
          otherViewport,
          line[3],
          line[4],
        ];
        newRtpoints.push(rotHandleWorldOne, rotHandleWorldTwo);

        const slabThicknessHandlesActive =
          data.handles.activeOperation === OPERATION.SLAB;
        const slabThicknessHandles = [line[11], line[12], line[13], line[14]];

        if (!minimalCrosshairConfig.enabled) {
          const slabThicknessHandleWorldOne = [
            viewport.canvasToWorld(line[11]),
            otherViewport,
            line[5],
            line[6],
          ];
          const slabThicknessHandleWorldTwo = [
            viewport.canvasToWorld(line[12]),
            otherViewport,
            line[5],
            line[6],
          ];
          const slabThicknessHandleWorldThree = [
            viewport.canvasToWorld(line[13]),
            otherViewport,
            line[7],
            line[8],
          ];
          const slabThicknessHandleWorldFour = [
            viewport.canvasToWorld(line[14]),
            otherViewport,
            line[7],
            line[8],
          ];
          newStpoints.push(
            slabThicknessHandleWorldOne,
            slabThicknessHandleWorldTwo,
            slabThicknessHandleWorldThree,
            slabThicknessHandleWorldFour
          );
        }

        let handleRadius =
          this.configuration.handleRadius *
          (this.configuration.enableHDPIHandles ? window.devicePixelRatio : 1);
        let opacity = 1;
        if (this.configuration.mobile?.enabled) {
          handleRadius = this.configuration.mobile.handleRadius;
          opacity = this.configuration.mobile.opacity;
        }

        if (
          (lineActive || this.configuration.mobile?.enabled) &&
          !rotHandlesActive &&
          !slabThicknessHandlesActive &&
          viewportDraggableRotatable &&
          viewportSlabThicknessControlsOn
        ) {
          // draw all handles inactive (rotation and slab thickness)
          let handleUID = `${lineIndex}One`;
          drawHandlesSvg(
            svgDrawingHelper,
            annotationUID,
            handleUID,
            rotationHandles,
            {
              color,
              handleRadius,
              opacity,
              type: 'circle',
            }
          );
          handleUID = `${lineIndex}Two`;
          drawHandlesSvg(
            svgDrawingHelper,
            annotationUID,
            handleUID,
            slabThicknessHandles,
            {
              color,
              handleRadius,
              opacity,
              type: 'rect',
            }
          );
        } else if (
          lineActive &&
          !rotHandlesActive &&
          !slabThicknessHandlesActive &&
          viewportDraggableRotatable
        ) {
          const handleUID = `${lineIndex}`;
          // draw rotation handles inactive
          drawHandlesSvg(
            svgDrawingHelper,
            annotationUID,
            handleUID,
            rotationHandles,
            {
              color,
              handleRadius,
              opacity,
              type: 'circle',
            }
          );
        } else if (
          selectedViewportId &&
          !rotHandlesActive &&
          !slabThicknessHandlesActive &&
          viewportSlabThicknessControlsOn
        ) {
          const handleUID = `${lineIndex}`;
          // draw slab thickness handles inactive
          drawHandlesSvg(
            svgDrawingHelper,
            annotationUID,
            handleUID,
            slabThicknessHandles,
            {
              color,
              handleRadius,
              opacity,
              type: 'rect',
            }
          );
        } else if (rotHandlesActive && viewportDraggableRotatable) {
          const handleUID = `${lineIndex}`;
          const handleRadius =
            this.configuration.handleRadius *
            (this.configuration.enableHDPIHandles
              ? window.devicePixelRatio
              : 1);
          // draw all rotation handles as active
          drawHandlesSvg(
            svgDrawingHelper,
            annotationUID,
            handleUID,
            rotationHandles,
            {
              color,
              handleRadius,
              fill: color,
              type: 'circle',
            }
          );
        } else if (
          slabThicknessHandlesActive &&
          selectedViewportId &&
          viewportSlabThicknessControlsOn
        ) {
          const handleRadius =
            this.configuration.handleRadius *
            (this.configuration.enableHDPIHandles
              ? window.devicePixelRatio
              : 1);
          // draw only the slab thickness handles for the active viewport as active
          drawHandlesSvg(
            svgDrawingHelper,
            annotationUID,
            lineUID,
            slabThicknessHandles,
            {
              color,
              handleRadius,
              fill: color,
              type: 'rect',
            }
          );
        }
        const slabThicknessValue = getSlabThicknessOrDefault(otherViewport);
        if (slabThicknessValue > 0.5 && viewportSlabThicknessControlsOn) {
          // draw slab thickness reference lines
          lineUID = `${lineIndex}STOne`;
          drawLineSvg(
            svgDrawingHelper,
            annotationUID,
            lineUID,
            line[5],
            line[6],
            {
              color,
              width: 1,
              lineDash: [2, 3],
            }
          );

          lineUID = `${lineIndex}STTwo`;
          drawLineSvg(
            svgDrawingHelper,
            annotationUID,
            lineUID,
            line[7],
            line[8],
            {
              color,
              width: 1,
              lineDash: [2, 3],
            }
          );
        }
      }
    });

    renderStatus = true;

    // Save new handles points in annotation
    data.handles.rotationPoints = newRtpoints;
    data.handles.slabThicknessPoints = newStpoints;

    if (this.configuration.viewportIndicators) {
      const { viewportIndicatorsConfig } = this.configuration;

      const xOffset = viewportIndicatorsConfig?.xOffset || 0.95;
      const yOffset = viewportIndicatorsConfig?.yOffset || 0.05;
      const referenceColorCoordinates = [
        clientWidth * xOffset,
        clientHeight * yOffset,
      ];

      const circleRadius =
        viewportIndicatorsConfig?.circleRadius || canvasDiagonalLength * 0.01;

      const circleUID = '0';
      drawCircleSvg(
        svgDrawingHelper,
        annotationUID,
        circleUID,
        referenceColorCoordinates as Types.Point2,
        circleRadius,
        { color, fill: color }
      );
    }

    if (this.configuration.centerPoint?.enabled) {
      const defaultColor = 'rgba(255, 255, 0, 0.5)';
      const defaultSize = 2;
      const maxAllowedSize = 5;

      const centerPointColor =
        this.configuration.centerPoint.color || defaultColor;
      const centerPointSize = Math.min(
        this.configuration.centerPoint.size || defaultSize,
        maxAllowedSize
      );

      drawCircleSvg(
        svgDrawingHelper,
        annotationUID,
        'centerPoint',
        crosshairCenterCanvas as Types.Point2,
        centerPointSize,
        {
          color: centerPointColor,
          fill: centerPointColor,
        }
      );
    }

    if (this.configuration.centerHandle?.enabled) {
      const centerHandleConfig = this.configuration.centerHandle;
      const centerHandleColor = centerHandleConfig.color || color;
      const centerHandleRadius =
        centerHandleConfig.radius ?? this.configuration.handleRadius ?? 3;

      const isCenterHandleActive =
        data.handles.activeOperation === OPERATION.DRAG &&
        data.activeViewportIds?.length > 0;

      drawCircleSvg(
        svgDrawingHelper,
        annotationUID,
        'centerHandle',
        crosshairCenterCanvas as Types.Point2,
        centerHandleRadius,
        {
          color: centerHandleColor,
          fill: isCenterHandleActive ? centerHandleColor : 'rgba(0, 0, 0, 0)',
          fillOpacity: isCenterHandleActive ? 0.6 : 0,
          strokeOpacity:
            viewportAnnotation.highlighted || isCenterHandleActive ? 1 : 0.8,
        }
      );
    }

    return renderStatus;
  };

  _getAnnotations = (enabledElement: Types.IEnabledElement) => {
    const { viewport } = enabledElement;
    const annotations =
      getAnnotations(this.getToolName(), viewport.element) || [];
    const viewportIds = this._getViewportsInfo().map(
      ({ viewportId }) => viewportId
    );

    // filter the annotations to only keep that are for this toolGroup
    const toolGroupAnnotations = annotations.filter((annotation) => {
      const { data } = annotation;
      return viewportIds.includes(data.viewportId);
    });

    return toolGroupAnnotations;
  };

  _onNewVolume = (_evt?: Event) => {
    this._syncVolumeListenersWithToolGroup();

    const forUID = this._getOwnFrameOfReferenceUID();
    const manager = getWorldPointManager();
    const sharedCenter = forUID ? manager.getPoint(forUID) : null;
    if (sharedCenter) {
      this.setToolCenter(sharedCenter, true);
      return;
    }

    if (this._lastValidToolCenter) {
      this.setToolCenter(this._lastValidToolCenter, false);
      return;
    }

    this._recomputeToolCenterFromAbsoluteCameras({
      emitEvent: true,
      updateViewportCameras: false,
    });
  };

  _autoPanViewportIfNecessary(
    viewportId: string,
    renderingEngine: Types.IRenderingEngine
  ): void {
    // 1. Check if the toolCenter is outside the viewport
    // 2. If it is outside, pan the viewport to fit in the toolCenter

    const viewport = renderingEngine.getViewport(viewportId);
    // Auto-pan is an in-plane focal-point shift (setCamera) that native
    // PLANAR_NEXT can't express; skip on native (comfort feature only).
    if (csUtils.isGenericViewport(viewport)) {
      return;
    }
    const { clientWidth, clientHeight } = getDisplayedCanvasSize(viewport);

    const toolCenterCanvas = viewport.worldToCanvas(this.toolCenter);

    // pan the viewport to fit the toolCenter in the direction
    // that is out of bounds
    const pan = this.configuration.autoPan.panSize;

    const visiblePointCanvas = <Types.Point2>[
      toolCenterCanvas[0],
      toolCenterCanvas[1],
    ];

    if (toolCenterCanvas[0] < 0) {
      visiblePointCanvas[0] = pan;
    } else if (toolCenterCanvas[0] > clientWidth) {
      visiblePointCanvas[0] = clientWidth - pan;
    }

    if (toolCenterCanvas[1] < 0) {
      visiblePointCanvas[1] = pan;
    } else if (toolCenterCanvas[1] > clientHeight) {
      visiblePointCanvas[1] = clientHeight - pan;
    }

    if (
      visiblePointCanvas[0] === toolCenterCanvas[0] &&
      visiblePointCanvas[1] === toolCenterCanvas[1]
    ) {
      return;
    }

    const visiblePointWorld = viewport.canvasToWorld(visiblePointCanvas);

    const deltaPointsWorld = [
      visiblePointWorld[0] - this.toolCenter[0],
      visiblePointWorld[1] - this.toolCenter[1],
      visiblePointWorld[2] - this.toolCenter[2],
    ];

    const camera = getViewportICamera(viewport);
    const { focalPoint, position } = camera;

    const updatedPosition = <Types.Point3>[
      position[0] - deltaPointsWorld[0],
      position[1] - deltaPointsWorld[1],
      position[2] - deltaPointsWorld[2],
    ];

    const updatedFocalPoint = <Types.Point3>[
      focalPoint[0] - deltaPointsWorld[0],
      focalPoint[1] - deltaPointsWorld[1],
      focalPoint[2] - deltaPointsWorld[2],
    ];

    const previousIgnoreFiredEvents = this._ignoreFiredEvents;
    this._ignoreFiredEvents = true;
    try {
      viewport.setCamera({
        focalPoint: updatedFocalPoint,
        position: updatedPosition,
      });
    } finally {
      this._ignoreFiredEvents = previousIgnoreFiredEvents;
    }

    viewport.render();
  }

  _areViewportIdArraysEqual = (viewportIdArrayOne, viewportIdArrayTwo) => {
    if (viewportIdArrayOne.length !== viewportIdArrayTwo.length) {
      return false;
    }

    for (let index = 0; index < viewportIdArrayOne.length; index++) {
      const id = viewportIdArrayOne[index];
      let itemFound = false;
      for (let i = 0; i < viewportIdArrayTwo.length; ++i) {
        if (id === viewportIdArrayTwo[i]) {
          itemFound = true;
          break;
        }
      }
      if (itemFound === false) {
        return false;
      }
    }

    return true;
  };

  // It filters the viewports with crosshairs and only return viewports
  // that have different camera.
  _getAnnotationsForViewportsWithDifferentCameras = (
    enabledElement,
    annotations
  ) => {
    const { viewportId, renderingEngine, viewport } = enabledElement;

    const otherViewportAnnotations = annotations.filter(
      (annotation) => annotation.data.viewportId !== viewportId
    );

    if (!otherViewportAnnotations || !otherViewportAnnotations.length) {
      return [];
    }

    const camera = getViewportICamera(viewport);
    const { viewPlaneNormal, position } = camera;

    const viewportsWithDifferentCameras = otherViewportAnnotations.filter(
      (annotation) => {
        const { viewportId } = annotation.data;
        const targetViewport = renderingEngine.getViewport(viewportId);
        const cameraOfTarget = getViewportICamera(targetViewport);

        return !(
          csUtils.isEqual(
            cameraOfTarget.viewPlaneNormal,
            viewPlaneNormal,
            1e-2
          ) && csUtils.isEqual(cameraOfTarget.position, position, 1)
        );
      }
    );

    return viewportsWithDifferentCameras;
  };

  _filterViewportWithSameOrientation = (
    enabledElement,
    referenceAnnotation,
    annotations
  ) => {
    const { renderingEngine } = enabledElement;
    const { data } = referenceAnnotation;
    const viewport = renderingEngine.getViewport(data.viewportId);

    const linkedViewportAnnotations = annotations.filter((annotation) => {
      const { data } = annotation;
      const otherViewport = renderingEngine.getViewport(data.viewportId);
      const otherViewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );

      return otherViewportControllable === true;
    });

    if (!linkedViewportAnnotations || !linkedViewportAnnotations.length) {
      return [];
    }

    const camera = getViewportICamera(viewport);
    const viewPlaneNormal = camera.viewPlaneNormal;
    vtkMath.normalize(viewPlaneNormal);

    const otherViewportsAnnotationsWithSameCameraDirection =
      linkedViewportAnnotations.filter((annotation) => {
        const { viewportId } = annotation.data;
        const otherViewport = renderingEngine.getViewport(viewportId);
        const otherCamera = getViewportICamera(otherViewport);
        const otherViewPlaneNormal = otherCamera.viewPlaneNormal;
        vtkMath.normalize(otherViewPlaneNormal);

        return (
          csUtils.isEqual(viewPlaneNormal, otherViewPlaneNormal, 1e-2) &&
          csUtils.isEqual(camera.viewUp, otherCamera.viewUp, 1e-2)
        );
      });

    return otherViewportsAnnotationsWithSameCameraDirection;
  };

  _filterAnnotationsByUniqueViewportOrientations = (
    enabledElement,
    annotations
  ) => {
    const { renderingEngine, viewport } = enabledElement;
    const camera = getViewportICamera(viewport);
    const viewPlaneNormal = camera.viewPlaneNormal;
    vtkMath.normalize(viewPlaneNormal);

    const otherLinkedViewportAnnotationsFromSameScene = annotations.filter(
      (annotation) => {
        const { data } = annotation;
        const otherViewport = renderingEngine.getViewport(data.viewportId);
        const otherViewportControllable = this._getReferenceLineControllable(
          otherViewport.id
        );

        return (
          viewport !== otherViewport &&
          // scene === otherScene &&
          otherViewportControllable === true
        );
      }
    );

    const otherViewportsAnnotationsWithUniqueCameras = [];
    // Iterate first on other viewport from the same scene linked
    for (
      let i = 0;
      i < otherLinkedViewportAnnotationsFromSameScene.length;
      ++i
    ) {
      const annotation = otherLinkedViewportAnnotationsFromSameScene[i];
      const { viewportId } = annotation.data;
      const otherViewport = renderingEngine.getViewport(viewportId);
      const otherCamera = getViewportICamera(otherViewport);
      const otherViewPlaneNormal = otherCamera.viewPlaneNormal;
      vtkMath.normalize(otherViewPlaneNormal);

      if (
        csUtils.isEqual(viewPlaneNormal, otherViewPlaneNormal, 1e-2) ||
        csUtils.isOpposite(viewPlaneNormal, otherViewPlaneNormal, 1e-2)
      ) {
        continue;
      }

      let cameraFound = false;
      for (
        let jj = 0;
        jj < otherViewportsAnnotationsWithUniqueCameras.length;
        ++jj
      ) {
        const annotation = otherViewportsAnnotationsWithUniqueCameras[jj];
        const { viewportId } = annotation.data;
        const stockedViewport = renderingEngine.getViewport(viewportId);
        const cameraOfStocked = getViewportICamera(stockedViewport);

        if (
          csUtils.isEqual(
            cameraOfStocked.viewPlaneNormal,
            otherCamera.viewPlaneNormal,
            1e-2
          ) &&
          csUtils.isEqual(cameraOfStocked.position, otherCamera.position, 1)
        ) {
          cameraFound = true;
        }
      }

      if (!cameraFound) {
        otherViewportsAnnotationsWithUniqueCameras.push(annotation);
      }
    }

    const otherNonLinkedViewportAnnotationsFromSameScene = annotations.filter(
      (annotation) => {
        const { data } = annotation;
        const otherViewport = renderingEngine.getViewport(data.viewportId);
        const otherViewportControllable = this._getReferenceLineControllable(
          otherViewport.id
        );

        return (
          viewport !== otherViewport &&
          // scene === otherScene &&
          otherViewportControllable !== true
        );
      }
    );

    // Iterate second on other viewport from the same scene non linked
    for (
      let i = 0;
      i < otherNonLinkedViewportAnnotationsFromSameScene.length;
      ++i
    ) {
      const annotation = otherNonLinkedViewportAnnotationsFromSameScene[i];
      const { viewportId } = annotation.data;
      const otherViewport = renderingEngine.getViewport(viewportId);

      const otherCamera = getViewportICamera(otherViewport);
      const otherViewPlaneNormal = otherCamera.viewPlaneNormal;
      vtkMath.normalize(otherViewPlaneNormal);

      if (
        csUtils.isEqual(viewPlaneNormal, otherViewPlaneNormal, 1e-2) ||
        csUtils.isOpposite(viewPlaneNormal, otherViewPlaneNormal, 1e-2)
      ) {
        continue;
      }

      let cameraFound = false;
      for (
        let jj = 0;
        jj < otherViewportsAnnotationsWithUniqueCameras.length;
        ++jj
      ) {
        const annotation = otherViewportsAnnotationsWithUniqueCameras[jj];
        const { viewportId } = annotation.data;
        const stockedViewport = renderingEngine.getViewport(viewportId);
        const cameraOfStocked = getViewportICamera(stockedViewport);

        if (
          csUtils.isEqual(
            cameraOfStocked.viewPlaneNormal,
            otherCamera.viewPlaneNormal,
            1e-2
          ) &&
          csUtils.isEqual(cameraOfStocked.position, otherCamera.position, 1)
        ) {
          cameraFound = true;
        }
      }

      if (!cameraFound) {
        otherViewportsAnnotationsWithUniqueCameras.push(annotation);
      }
    }

    // Iterate on all the viewport
    const otherViewportAnnotations =
      this._getAnnotationsForViewportsWithDifferentCameras(
        enabledElement,
        annotations
      );

    for (let i = 0; i < otherViewportAnnotations.length; ++i) {
      const annotation = otherViewportAnnotations[i];
      if (
        otherViewportsAnnotationsWithUniqueCameras.some(
          (element) => element === annotation
        )
      ) {
        continue;
      }

      const { viewportId } = annotation.data;
      const otherViewport = renderingEngine.getViewport(viewportId);
      const otherCamera = getViewportICamera(otherViewport);
      const otherViewPlaneNormal = otherCamera.viewPlaneNormal;
      vtkMath.normalize(otherViewPlaneNormal);

      if (
        csUtils.isEqual(viewPlaneNormal, otherViewPlaneNormal, 1e-2) ||
        csUtils.isOpposite(viewPlaneNormal, otherViewPlaneNormal, 1e-2)
      ) {
        continue;
      }

      let cameraFound = false;
      for (
        let jj = 0;
        jj < otherViewportsAnnotationsWithUniqueCameras.length;
        ++jj
      ) {
        const annotation = otherViewportsAnnotationsWithUniqueCameras[jj];
        const { viewportId } = annotation.data;
        const stockedViewport = renderingEngine.getViewport(viewportId);
        const cameraOfStocked = getViewportICamera(stockedViewport);

        if (
          csUtils.isEqual(
            cameraOfStocked.viewPlaneNormal,
            otherCamera.viewPlaneNormal,
            1e-2
          ) &&
          csUtils.isEqual(cameraOfStocked.position, otherCamera.position, 1)
        ) {
          cameraFound = true;
        }
      }

      if (!cameraFound) {
        otherViewportsAnnotationsWithUniqueCameras.push(annotation);
      }
    }

    return otherViewportsAnnotationsWithUniqueCameras;
  };

  _checkIfViewportsRenderingSameScene = (viewport, otherViewport) => {
    // Native PLANAR_NEXT has no getAllVolumeIds; compare view-reference dataId
    // instead (MPR planes of one volume share a dataId).
    if (
      csUtils.isGenericViewport(viewport) ||
      csUtils.isGenericViewport(otherViewport)
    ) {
      const ref = viewport.getViewReference?.();
      const otherRef = otherViewport.getViewReference?.();
      const a = ref?.dataId ?? ref?.FrameOfReferenceUID;
      const b = otherRef?.dataId ?? otherRef?.FrameOfReferenceUID;
      return !!a && a === b;
    }

    const volumeIds = viewport.getAllVolumeIds();
    const otherVolumeIds = otherViewport.getAllVolumeIds();

    return (
      volumeIds.length === otherVolumeIds.length &&
      volumeIds.every((id) => otherVolumeIds.includes(id))
    );
  };

  _jump = (enabledElement, jumpWorld) => {
    state.isInteractingWithTool = true;
    const { viewport, renderingEngine } = enabledElement;

    const annotations = this._getAnnotations(enabledElement);

    const delta: Types.Point3 = [0, 0, 0];
    vtkMath.subtract(jumpWorld, this.toolCenter, delta);

    // TRANSLATION
    // get the annotation of the other viewport which are parallel to the delta shift and are of the same scene
    const otherViewportAnnotations =
      this._getAnnotationsForViewportsWithDifferentCameras(
        enabledElement,
        annotations
      );

    const viewportsAnnotationsToUpdate = otherViewportAnnotations.filter(
      (annotation) => {
        const { data } = annotation;
        const otherViewport = renderingEngine.getViewport(data.viewportId);

        const sameScene = this._checkIfViewportsRenderingSameScene(
          viewport,
          otherViewport
        );

        return (
          this._getReferenceLineControllable(otherViewport.id) &&
          this._getReferenceLineDraggableRotatable(otherViewport.id) &&
          sameScene
        );
      }
    );

    if (viewportsAnnotationsToUpdate.length === 0) {
      state.isInteractingWithTool = false;
      return false;
    }

    this._applyDeltaShiftToSelectedViewportCameras(
      renderingEngine,
      viewportsAnnotationsToUpdate,
      delta
    );

    const forUID = this._getOwnFrameOfReferenceUID();
    const manager = getWorldPointManager();
    if (forUID) {
      manager.setPoint(forUID, jumpWorld, viewport.id, renderingEngine.id);
    }
    this._commitToolCenter(jumpWorld, {
      senderViewportId: viewport.id,
      senderRenderingEngineId: renderingEngine.id,
    });

    triggerEvent(eventTarget, Events.CROSSHAIR_TOOL_CENTER_CHANGED, {
      toolGroupId: this.toolGroupId,
      toolCenter: forUID ? (manager.getPoint(forUID) ?? jumpWorld) : jumpWorld,
      FrameOfReferenceUID: forUID,
    });

    viewport.render();

    state.isInteractingWithTool = false;

    return true;
  };

  _activateModify = (element) => {
    this._syncVolumeListenersWithToolGroup();
    this._recomputeToolCenterFromAbsoluteCameras({
      emitEvent: false,
      updateViewportCameras: false,
    });

    // mobile sometimes has lingering interaction even when touchEnd triggers
    // this check allows for multiple handles to be active which doesn't affect
    // tool usage.
    state.isInteractingWithTool = !this.configuration.mobile?.enabled;

    element.addEventListener(Events.MOUSE_UP, this._endCallback);
    element.addEventListener(Events.MOUSE_DRAG, this._dragCallback);
    element.addEventListener(Events.MOUSE_CLICK, this._endCallback);

    element.addEventListener(Events.TOUCH_END, this._endCallback);
    element.addEventListener(Events.TOUCH_DRAG, this._dragCallback);
    element.addEventListener(Events.TOUCH_TAP, this._endCallback);
  };

  _deactivateModify = (element) => {
    state.isInteractingWithTool = false;

    element.removeEventListener(Events.MOUSE_UP, this._endCallback);
    element.removeEventListener(Events.MOUSE_DRAG, this._dragCallback);
    element.removeEventListener(Events.MOUSE_CLICK, this._endCallback);

    element.removeEventListener(Events.TOUCH_END, this._endCallback);
    element.removeEventListener(Events.TOUCH_DRAG, this._dragCallback);
    element.removeEventListener(Events.TOUCH_TAP, this._endCallback);
  };

  _endCallback = (evt: EventTypes.InteractionEventType) => {
    const eventDetail = evt.detail;
    const { element } = eventDetail;

    // Capture the operation type BEFORE it is cleared below — we need it to
    // decide whether a post-interaction camera-plane recompute is allowed.
    const lastOperation =
      this.editData?.annotation?.data?.handles?.activeOperation;
    const wasDragOperation = lastOperation === OPERATION.DRAG;

    if (this.editData?.annotation?.data) {
      this.editData.annotation.data.handles.activeOperation = null;
      this.editData.annotation.data.activeViewportIds = [];
    }

    this._deactivateModify(element);

    // Snap any StackViewport(s) to the slice closest to the final tool center
    // only now that the drag has ended (during the drag the marker moves
    // smoothly on canvas without switching slices — see `_dragCallback` — to
    // avoid a slice-change/recompute feedback loop).
    const forUID = this._getOwnFrameOfReferenceUID();
    const manager = getWorldPointManager();
    const point = forUID ? manager.getPoint(forUID) : null;
    if (point) {
      this._syncStackSlices(this._getViewportsInfo(), point);
    }

    // A DRAG only translates cameras (no orientation change), so toolCenter is
    // already correct — recomputing from camera-plane intersection would make
    // it snap/drift on release. Only recompute after ROTATE/SLAB.
    if (!wasDragOperation) {
      this._recomputeToolCenterFromAbsoluteCameras({
        emitEvent: true,
        updateViewportCameras: false,
      });
    }

    resetElementCursor(element);

    this.editData = null;

    const requireSameOrientation = false;
    const viewportIdsToRender = getViewportIdsWithToolToRender(
      element,
      this.getToolName(),
      requireSameOrientation
    );

    triggerAnnotationRenderForViewportIds(viewportIdsToRender);
  };

  _dragCallback = (evt: EventTypes.InteractionEventType) => {
    const eventDetail = evt.detail;
    const delta = eventDetail.deltaPoints.world;

    if (
      Math.abs(delta[0]) < 1e-3 &&
      Math.abs(delta[1]) < 1e-3 &&
      Math.abs(delta[2]) < 1e-3
    ) {
      return;
    }

    const { element } = eventDetail;
    const enabledElement = getEnabledElement(element);
    const { renderingEngine, viewport } = enabledElement;
    const annotations = this._getAnnotations(
      enabledElement
    ) as CrosshairsAnnotation[];
    const filteredToolAnnotations =
      this.filterInteractableAnnotationsForElement(element, annotations);

    // viewport Annotation
    const viewportAnnotation = filteredToolAnnotations[0];
    if (!viewportAnnotation) {
      return;
    }

    const { handles } = viewportAnnotation.data;
    const { currentPoints } = evt.detail;
    const canvasCoords = currentPoints.canvas;

    if (handles.activeOperation === OPERATION.DRAG) {
      const forUID = this._getOwnFrameOfReferenceUID();
      const manager = getWorldPointManager();

      // For a StackViewport, snap the point directly to
      // canvasToWorld(currentCanvas) instead of accumulating an in-plane delta
      // (which drifts off the slice plane over time). Guarantees 1:1 mouse
      // tracking on the physical slice plane.
      const isStackSource = viewport instanceof StackViewport;

      // World-space movement of this drag step, used to scroll the other MPR
      // reference planes. Raw `delta` for an MPR source; derived from the
      // snapped point movement for a StackViewport.
      let actualDelta: Types.Point3 = delta as Types.Point3;

      let newToolCenter: Types.Point3;
      if (isStackSource) {
        const currentWorld = viewport.canvasToWorld(
          currentPoints.canvas
        ) as Types.Point3;

        newToolCenter = [currentWorld[0], currentWorld[1], currentWorld[2]];

        actualDelta = [
          newToolCenter[0] - this.toolCenter[0],
          newToolCenter[1] - this.toolCenter[1],
          newToolCenter[2] - this.toolCenter[2],
        ];

        this._debugLog('_dragCallback:DRAG:stack-source', {
          viewportId: viewport.id,
          currentWorld: this._formatPoint3(currentWorld),
          actualDelta: this._formatPoint3(actualDelta),
          newToolCenter: this._formatPoint3(newToolCenter),
        });
      } else {
        newToolCenter = [
          this.toolCenter[0] + delta[0],
          this.toolCenter[1] + delta[1],
          this.toolCenter[2] + delta[2],
        ];

        this._debugLog('_dragCallback:DRAG:volume-source', {
          viewportId: viewport.id,
          delta: this._formatPoint3(delta as Types.Point3),
          newToolCenter: this._formatPoint3(newToolCenter),
        });
      }

      // Scroll the other MPR reformat planes along their normals so they
      // follow the drag. StackViewports are NOT slice-switched here (that would
      // cause a feedback loop); their slice snaps on drag end (_endCallback).
      const otherViewportAnnotations =
        this._getAnnotationsForViewportsWithDifferentCameras(
          enabledElement,
          annotations
        );

      const viewportsAnnotationsToUpdate = otherViewportAnnotations.filter(
        (annotation) => {
          const { data } = annotation;
          const otherViewport = renderingEngine.getViewport(data.viewportId);
          const otherViewportControllable =
            this._getReferenceLineControllable(otherViewport.id);
          const otherViewportDraggableRotatable =
            this._getReferenceLineDraggableRotatable(otherViewport.id);

          return (
            !(otherViewport instanceof StackViewport) &&
            otherViewportControllable === true &&
            otherViewportDraggableRotatable === true &&
            viewportAnnotation.data.activeViewportIds.find(
              (id) => id === otherViewport.id
            )
          );
        }
      );

      // Use the in-plane `actualDelta` (exact world movement in the slice
      // plane) so MPR reference planes follow the mouse precisely. For MPR
      // sources `actualDelta === delta`.
      this._applyDeltaShiftToSelectedViewportCameras(
        renderingEngine,
        viewportsAnnotationsToUpdate,
        actualDelta
      );

      // Update the shared point in the manager (source of truth). The manager
      // notifies every *other* subscriber (excluding this sender viewport), so
      // the cursor never gets snapped back to its original position.
      this._commitToolCenter(newToolCenter, {
        senderViewportId: viewport.id,
        senderRenderingEngineId: renderingEngine.id,
      });

      triggerEvent(eventTarget, Events.CROSSHAIR_TOOL_CENTER_CHANGED, {
        toolGroupId: this.toolGroupId,
        toolCenter: forUID ? (manager.getPoint(forUID) ?? newToolCenter) : newToolCenter,
        FrameOfReferenceUID: forUID,
      });
    } else if (handles.activeOperation === OPERATION.ROTATE) {
      // ROTATION — skipped on native PLANAR_NEXT (no setCamera / oblique
      // orientation write); the rotate handles are inert rather than throwing.
      if (csUtils.isGenericViewport(enabledElement.viewport)) {
        return;
      }
      const otherViewportAnnotations =
        this._getAnnotationsForViewportsWithDifferentCameras(
          enabledElement,
          annotations
        );

      const viewportsAnnotationsToUpdate = otherViewportAnnotations.filter(
        (annotation) => {
          const { data } = annotation;
          const otherViewport = renderingEngine.getViewport(data.viewportId);
          const otherViewportControllable = this._getReferenceLineControllable(
            otherViewport.id
          );
          const otherViewportDraggableRotatable =
            this._getReferenceLineDraggableRotatable(otherViewport.id);

          return (
            otherViewportControllable === true &&
            otherViewportDraggableRotatable === true
          );
        }
      );

      const dir1 = vec2.create();
      const dir2 = vec2.create();

      const center: Types.Point3 = [
        this.toolCenter[0],
        this.toolCenter[1],
        this.toolCenter[2],
      ];

      const centerCanvas = viewport.worldToCanvas(center);

      const finalPointCanvas = eventDetail.currentPoints.canvas;
      const originalPointCanvas = vec2.create();
      vec2.sub(
        originalPointCanvas,
        finalPointCanvas,
        eventDetail.deltaPoints.canvas
      );
      vec2.sub(dir1, originalPointCanvas, <vec2>centerCanvas);
      vec2.sub(dir2, finalPointCanvas, <vec2>centerCanvas);

      let angle = vec2.angle(dir1, dir2);

      if (
        this._isClockWise(centerCanvas, originalPointCanvas, finalPointCanvas)
      ) {
        angle *= -1;
      }

      // Rounding the angle to allow rotated handles to be undone
      // If we don't round and rotate handles clockwise by 0.0131233 radians,
      // there's no assurance that the counter-clockwise rotation occurs at
      // precisely -0.0131233, resulting in the drawn annotations being lost.
      angle = Math.round(angle * 100) / 100;

      const rotationAxis = getViewportICamera(viewport).viewPlaneNormal;
      // @ts-ignore : vtkjs incorrect typing
      const { matrix } = vtkMatrixBuilder
        .buildFromRadian()
        .translate(center[0], center[1], center[2])
        // @ts-ignore
        .rotate(angle, rotationAxis) //todo: why we are passing
        .translate(-center[0], -center[1], -center[2]);

      const otherViewportsIds = [];
      // update camera for the other viewports.
      // NOTE: The lines then are rendered by the onCameraModified
      const previousIgnoreFiredEvents = this._ignoreFiredEvents;
      this._ignoreFiredEvents = true;
      try {
        viewportsAnnotationsToUpdate.forEach((annotation) => {
          const { data } = annotation;
          data.handles.toolCenter = center;

          const otherViewport = renderingEngine.getViewport(data.viewportId);
          const camera = getViewportICamera(otherViewport);
          const { viewUp, position, focalPoint } = camera;

          viewUp[0] += position[0];
          viewUp[1] += position[1];
          viewUp[2] += position[2];

          vec3.transformMat4(focalPoint, focalPoint, matrix);
          vec3.transformMat4(position, position, matrix);
          vec3.transformMat4(viewUp, viewUp, matrix);

          viewUp[0] -= position[0];
          viewUp[1] -= position[1];
          viewUp[2] -= position[2];

          otherViewport.setCamera({
            position,
            viewUp,
            focalPoint,
          });
          otherViewportsIds.push(otherViewport.id);
        });
      } finally {
        this._ignoreFiredEvents = previousIgnoreFiredEvents;
      }
      renderingEngine.renderViewports(otherViewportsIds);
      this._recomputeToolCenterFromAbsoluteCameras({
        emitEvent: true,
        updateViewportCameras: false,
      });
    } else if (handles.activeOperation === OPERATION.SLAB) {
      // SLAB THICKNESS — skipped on native PLANAR_NEXT (no slab-thickness API).
      if (csUtils.isGenericViewport(enabledElement.viewport)) {
        return;
      }
      // this should be just the active one under the mouse,
      const otherViewportAnnotations =
        this._getAnnotationsForViewportsWithDifferentCameras(
          enabledElement,
          annotations
        );

      const referenceAnnotations = otherViewportAnnotations.filter(
        (annotation) => {
          const { data } = annotation;
          const otherViewport = renderingEngine.getViewport(data.viewportId);
          const otherViewportControllable = this._getReferenceLineControllable(
            otherViewport.id
          );
          const otherViewportSlabThicknessControlsOn =
            this._getReferenceLineSlabThicknessControlsOn(otherViewport.id);

          return (
            otherViewportControllable === true &&
            otherViewportSlabThicknessControlsOn === true &&
            viewportAnnotation.data.activeViewportIds.find(
              (id) => id === otherViewport.id
            )
          );
        }
      );

      if (referenceAnnotations.length === 0) {
        return;
      }
      const viewportsAnnotationsToUpdate =
        this._filterViewportWithSameOrientation(
          enabledElement,
          referenceAnnotations[0],
          annotations
        );

      const viewportsIds = [];
      viewportsIds.push(viewport.id);
      viewportsAnnotationsToUpdate.forEach(
        (annotation: CrosshairsAnnotation) => {
          const { data } = annotation;

          const otherViewport = renderingEngine.getViewport(
            data.viewportId
          ) as Types.IVolumeViewport;
          const camera = getViewportICamera(otherViewport);
          const normal = camera.viewPlaneNormal;

          const dotProd = vtkMath.dot(delta, normal);
          const projectedDelta: Types.Point3 = [...normal];
          vtkMath.multiplyScalar(projectedDelta, dotProd);

          if (
            Math.abs(projectedDelta[0]) > 1e-3 ||
            Math.abs(projectedDelta[1]) > 1e-3 ||
            Math.abs(projectedDelta[2]) > 1e-3
          ) {
            const mod = Math.sqrt(
              projectedDelta[0] * projectedDelta[0] +
                projectedDelta[1] * projectedDelta[1] +
                projectedDelta[2] * projectedDelta[2]
            );

            const currentPoint = eventDetail.lastPoints.world;
            const direction: Types.Point3 = [0, 0, 0];

            const currentCenter: Types.Point3 = [
              this.toolCenter[0],
              this.toolCenter[1],
              this.toolCenter[2],
            ];

            // use this.toolCenter only if viewportDraggableRotatable
            const viewportDraggableRotatable =
              this._getReferenceLineDraggableRotatable(otherViewport.id);
            if (!viewportDraggableRotatable) {
              const { rotationPoints } = (<CrosshairsAnnotationData>(
                this.editData.annotation.data
              )).handles;
              // Todo: what is a point uid?
              const otherViewportRotationPoints = rotationPoints.filter(
                // @ts-expect-error
                (point) => point[1].uid === otherViewport.id
              );
              if (otherViewportRotationPoints.length === 2) {
                const point1 = viewport.canvasToWorld(
                  // @ts-expect-error
                  otherViewportRotationPoints[0][3]
                );
                const point2 = viewport.canvasToWorld(
                  // @ts-expect-error
                  otherViewportRotationPoints[1][3]
                );
                vtkMath.add(point1, point2, currentCenter);
                vtkMath.multiplyScalar(<Types.Point3>currentCenter, 0.5);
              }
            }

            vtkMath.subtract(currentPoint, currentCenter, direction);
            const dotProdDirection = vtkMath.dot(direction, normal);
            const projectedDirection: Types.Point3 = [...normal];
            vtkMath.multiplyScalar(projectedDirection, dotProdDirection);
            const normalizedProjectedDirection: Types.Point3 = [
              projectedDirection[0],
              projectedDirection[1],
              projectedDirection[2],
            ];
            vec3.normalize(
              normalizedProjectedDirection,
              normalizedProjectedDirection
            );
            const normalizedProjectedDelta: Types.Point3 = [
              projectedDelta[0],
              projectedDelta[1],
              projectedDelta[2],
            ];
            vec3.normalize(normalizedProjectedDelta, normalizedProjectedDelta);

            let slabThicknessValue = getSlabThicknessOrDefault(otherViewport);
            if (
              csUtils.isOpposite(
                normalizedProjectedDirection,
                normalizedProjectedDelta,
                1e-3
              )
            ) {
              slabThicknessValue -= mod;
            } else {
              slabThicknessValue += mod;
            }

            slabThicknessValue = Math.abs(slabThicknessValue);
            slabThicknessValue = Math.max(
              RENDERING_DEFAULTS.MINIMUM_SLAB_THICKNESS,
              slabThicknessValue
            );

            const near = this._pointNearReferenceLine(
              viewportAnnotation,
              canvasCoords,
              6,
              otherViewport
            );

            if (near) {
              slabThicknessValue = RENDERING_DEFAULTS.MINIMUM_SLAB_THICKNESS;
            }

            // We want to set the slabThickness for the viewport's actors but
            // since the crosshairs tool instance has configuration regarding which
            // actorUIDs (in case of volume -> actorUID = volumeIds) to set the
            // slabThickness for, we need to delegate the slabThickness setting
            // to the crosshairs tool instance of the toolGroup since configurations
            // exist on the toolInstance and each toolGroup has its own crosshairs
            // tool instance (Otherwise, we would need to set this filterActorUIDsToSetSlabThickness at
            // the viewport level which makes tool and viewport state convoluted).
            const toolGroup = getToolGroupForViewport(
              otherViewport.id,
              renderingEngine.id
            );
            const crosshairsInstance = toolGroup.getToolInstance(
              this.getToolName()
            );
            crosshairsInstance.setSlabThickness(
              otherViewport,
              slabThicknessValue
            );

            viewportsIds.push(otherViewport.id);
          }
        }
      );
      renderingEngine.renderViewports(viewportsIds);
      this._recomputeToolCenterFromAbsoluteCameras({
        emitEvent: true,
        updateViewportCameras: false,
      });
    }

    const requireSameOrientation = false;
    const viewportIdsToRender = getViewportIdsWithToolToRender(
      element,
      this.getToolName(),
      requireSameOrientation
    );

    triggerAnnotationRenderForViewportIds(viewportIdsToRender);
  };

  setSlabThickness(viewport, slabThickness) {
    // Native PLANAR_NEXT has no slab API; guard here for safety.
    if (csUtils.isGenericViewport(viewport)) {
      return;
    }
    let actorUIDs;
    const { filterActorUIDsToSetSlabThickness } = this.configuration;
    if (
      filterActorUIDsToSetSlabThickness &&
      filterActorUIDsToSetSlabThickness.length > 0
    ) {
      actorUIDs = filterActorUIDsToSetSlabThickness;
    }

    let blendModeToUse = this.configuration.slabThicknessBlendMode;
    if (slabThickness === RENDERING_DEFAULTS.MINIMUM_SLAB_THICKNESS) {
      blendModeToUse = Enums.BlendModes.COMPOSITE;
    }

    const immediate = false;
    viewport.setBlendMode(blendModeToUse, actorUIDs, immediate);
    viewport.setSlabThickness(slabThickness, actorUIDs);
  }

  _isClockWise(a, b, c) {
    // return true if the rotation is clockwise
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0;
  }

  _applyDeltaShiftToSelectedViewportCameras(
    renderingEngine,
    viewportsAnnotationsToUpdate,
    delta
  ) {
    // update camera for the other viewports.
    viewportsAnnotationsToUpdate.forEach((annotation) => {
      this._applyDeltaShiftToViewportCamera(renderingEngine, annotation, delta);
    });
  }

  _applyDeltaShiftToViewportCamera(
    renderingEngine: Types.IRenderingEngine,
    annotation,
    delta
  ) {
    // update camera for the other viewports.
    const { data } = annotation;

    const viewport = renderingEngine.getViewport(data.viewportId);

    // 3D volume renders have no per-slice scroll concept; their free-orbit
    // camera must not be shifted, only re-project the point on render.
    if (
      csUtils.isGenericViewport(viewport) ||
      viewport instanceof VolumeViewport3D
    ) {
      return;
    }

    const camera = getViewportICamera(viewport);
    const normal = camera.viewPlaneNormal;

    // Project delta over camera normal
    // (we don't need to pan, we need only to scroll the camera as in the wheel stack scroll tool)
    const dotProd = vtkMath.dot(delta, normal);
    const projectedDelta: Types.Point3 = [...normal];
    vtkMath.multiplyScalar(projectedDelta, dotProd);

    if (
      Math.abs(projectedDelta[0]) > 1e-3 ||
      Math.abs(projectedDelta[1]) > 1e-3 ||
      Math.abs(projectedDelta[2]) > 1e-3
    ) {
      const newFocalPoint: Types.Point3 = [0, 0, 0];
      const newPosition: Types.Point3 = [0, 0, 0];

      vtkMath.add(camera.focalPoint, projectedDelta, newFocalPoint);
      vtkMath.add(camera.position, projectedDelta, newPosition);

      const previousIgnoreFiredEvents = this._ignoreFiredEvents;
      this._ignoreFiredEvents = true;
      try {
        if (csUtils.isGenericViewport(viewport)) {
          // Native has no setCamera; navigate by view reference (snaps to slice).
          jumpToFocalPoint(viewport, newFocalPoint);
        } else {
          viewport.setCamera({
            focalPoint: newFocalPoint,
            position: newPosition,
          });
        }
      } finally {
        this._ignoreFiredEvents = previousIgnoreFiredEvents;
      }
      viewport.render();
    }
  }

  _pointNearReferenceLine = (
    annotation,
    canvasCoords,
    proximity,
    lineViewport
  ) => {
    const { data } = annotation;
    const { rotationPoints } = data.handles;

    for (let i = 0; i < rotationPoints.length - 1; ++i) {
      const otherViewport = rotationPoints[i][1];
      if (otherViewport.id !== lineViewport.id) {
        continue;
      }

      const viewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      if (!viewportControllable) {
        continue;
      }

      const lineSegment1 = {
        start: {
          x: rotationPoints[i][2][0],
          y: rotationPoints[i][2][1],
        },
        end: {
          x: rotationPoints[i][3][0],
          y: rotationPoints[i][3][1],
        },
      };

      const distanceToPoint1 = lineSegment.distanceToPoint(
        [lineSegment1.start.x, lineSegment1.start.y],
        [lineSegment1.end.x, lineSegment1.end.y],
        [canvasCoords[0], canvasCoords[1]]
      );

      const lineSegment2 = {
        start: {
          x: rotationPoints[i + 1][2][0],
          y: rotationPoints[i + 1][2][1],
        },
        end: {
          x: rotationPoints[i + 1][3][0],
          y: rotationPoints[i + 1][3][1],
        },
      };

      const distanceToPoint2 = lineSegment.distanceToPoint(
        [lineSegment2.start.x, lineSegment2.start.y],
        [lineSegment2.end.x, lineSegment2.end.y],
        [canvasCoords[0], canvasCoords[1]]
      );

      if (distanceToPoint1 <= proximity || distanceToPoint2 <= proximity) {
        return true;
      }

      // rotation handles are two for viewport
      i++;
    }

    return false;
  };

  _getRotationHandleNearImagePoint(
    viewport,
    annotation,
    canvasCoords,
    proximity
  ) {
    const minimalCrosshairConfig = getMinimalCrosshairConfig(
      this.configuration
    );
    const { data } = annotation;
    const { rotationPoints } = data.handles;

    for (let i = 0; i < rotationPoints.length; i++) {
      const point = rotationPoints[i][0];
      const otherViewport = rotationPoints[i][1];
      const viewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      if (!viewportControllable) {
        continue;
      }

      const viewportDraggableRotatable =
        !minimalCrosshairConfig.enabled &&
        this._getReferenceLineDraggableRotatable(otherViewport.id);
      if (!viewportDraggableRotatable) {
        continue;
      }

      const annotationCanvasCoordinate = viewport.worldToCanvas(point);
      if (vec2.distance(canvasCoords, annotationCanvasCoordinate) < proximity) {
        data.handles.activeOperation = OPERATION.ROTATE;

        this.editData = {
          annotation,
        };

        return point;
      }
    }

    return null;
  }

  _getSlabThicknessHandleNearImagePoint(
    viewport,
    annotation,
    canvasCoords,
    proximity
  ) {
    const minimalCrosshairConfig = getMinimalCrosshairConfig(
      this.configuration
    );
    const { data } = annotation;
    const { slabThicknessPoints } = data.handles;

    for (let i = 0; i < slabThicknessPoints.length; i++) {
      const point = slabThicknessPoints[i][0];
      const otherViewport = slabThicknessPoints[i][1];
      const viewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      if (!viewportControllable) {
        continue;
      }

      const viewportSlabThicknessControlsOn =
        !minimalCrosshairConfig.enabled &&
        this._getReferenceLineSlabThicknessControlsOn(otherViewport.id);
      if (!viewportSlabThicknessControlsOn) {
        continue;
      }

      const annotationCanvasCoordinate = viewport.worldToCanvas(point);
      if (vec2.distance(canvasCoords, annotationCanvasCoordinate) < proximity) {
        data.handles.activeOperation = OPERATION.SLAB;

        data.activeViewportIds = [otherViewport.id];

        this.editData = {
          annotation,
        };

        return point;
      }
    }

    return null;
  }

  /**
   * Returns the ids of every other controllable+draggable viewport in the
   * toolGroup, to translate together when dragging the center handle.
   */
  _getAllDraggableOtherViewportIds = (currentViewportId: string): string[] => {
    const viewportsInfo = this._getViewportsInfo();
    const viewportIds = [];

    viewportsInfo.forEach(({ viewportId }) => {
      if (viewportId === currentViewportId) {
        return;
      }

      if (
        this._getReferenceLineControllable(viewportId) &&
        this._getReferenceLineDraggableRotatable(viewportId)
      ) {
        viewportIds.push(viewportId);
      }
    });

    return viewportIds;
  };

  /**
   * If `canvasCoords` is near the center handle, marks the annotation for a
   * translation (DRAG) of every other draggable viewport.
   */
  _getCenterHandleNearImagePoint(
    viewport,
    annotation: CrosshairsAnnotation,
    canvasCoords: Types.Point2,
    proximity: number
  ): boolean {
    const centerHandleConfig = this.configuration.centerHandle;
    if (!centerHandleConfig?.enabled) {
      return false;
    }

    const centerCanvas = viewport.worldToCanvas(this.toolCenter);
    const centerHandleRadius =
      centerHandleConfig.radius ?? this.configuration.handleRadius ?? 3;
    const centerHandleProximity = Math.max(centerHandleRadius, proximity);

    if (vec2.distance(canvasCoords, centerCanvas) > centerHandleProximity) {
      return false;
    }

    const { data } = annotation;
    data.handles.activeOperation = OPERATION.DRAG;
    data.activeViewportIds = this._getAllDraggableOtherViewportIds(viewport.id);

    this.editData = {
      annotation,
    };

    return true;
  }

  _pointNearTool(element, annotation, canvasCoords, proximity) {
    const minimalCrosshairConfig = getMinimalCrosshairConfig(
      this.configuration
    );
    const enabledElement = getEnabledElement(element);
    const { viewport } = enabledElement;

    if (
      this._getCenterHandleNearImagePoint(
        viewport,
        annotation,
        canvasCoords,
        proximity
      )
    ) {
      return true;
    }

    const { clientWidth, clientHeight } = getDisplayedCanvasSize(viewport);
    const canvasDiagonalLength = Math.sqrt(
      clientWidth * clientWidth + clientHeight * clientHeight
    );
    const { data } = annotation;

    const { rotationPoints } = data.handles;
    const { slabThicknessPoints } = data.handles;
    const viewportIdArray = [];

    for (let i = 0; i < rotationPoints.length - 1; ++i) {
      const otherViewport = rotationPoints[i][1];
      const viewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      const viewportDraggableRotatable =
        this._getReferenceLineDraggableRotatable(otherViewport.id);

      if (
        !viewportControllable ||
        (!viewportDraggableRotatable && !minimalCrosshairConfig.enabled)
      ) {
        continue;
      }

      const lineSegment1 = {
        start: {
          x: rotationPoints[i][2][0],
          y: rotationPoints[i][2][1],
        },
        end: {
          x: rotationPoints[i][3][0],
          y: rotationPoints[i][3][1],
        },
      };

      const distanceToPoint1 = lineSegment.distanceToPoint(
        [lineSegment1.start.x, lineSegment1.start.y],
        [lineSegment1.end.x, lineSegment1.end.y],
        [canvasCoords[0], canvasCoords[1]]
      );

      const lineSegment2 = {
        start: {
          x: rotationPoints[i + 1][2][0],
          y: rotationPoints[i + 1][2][1],
        },
        end: {
          x: rotationPoints[i + 1][3][0],
          y: rotationPoints[i + 1][3][1],
        },
      };

      const distanceToPoint2 = lineSegment.distanceToPoint(
        [lineSegment2.start.x, lineSegment2.start.y],
        [lineSegment2.end.x, lineSegment2.end.y],
        [canvasCoords[0], canvasCoords[1]]
      );

      if (distanceToPoint1 <= proximity || distanceToPoint2 <= proximity) {
        viewportIdArray.push(otherViewport.id);
        data.handles.activeOperation = OPERATION.DRAG;
      }

      // rotation handles are two for viewport
      i++;
    }

    for (let i = 0; i < slabThicknessPoints.length - 1; ++i) {
      const otherViewport = slabThicknessPoints[i][1];
      if (viewportIdArray.find((id) => id === otherViewport.id)) {
        continue;
      }

      const viewportControllable = this._getReferenceLineControllable(
        otherViewport.id
      );
      const viewportSlabThicknessControlsOn =
        !minimalCrosshairConfig.enabled &&
        this._getReferenceLineSlabThicknessControlsOn(otherViewport.id);

      if (!viewportControllable || !viewportSlabThicknessControlsOn) {
        continue;
      }

      const stPointLineCanvas1 = slabThicknessPoints[i][2];
      const stPointLineCanvas2 = slabThicknessPoints[i][3];

      const centerCanvas = vec2.create();
      vec2.add(centerCanvas, stPointLineCanvas1, stPointLineCanvas2);
      vec2.scale(centerCanvas, centerCanvas, 0.5);

      const canvasUnitVectorFromCenter = vec2.create();
      vec2.subtract(
        canvasUnitVectorFromCenter,
        stPointLineCanvas1,
        centerCanvas
      );
      vec2.normalize(canvasUnitVectorFromCenter, canvasUnitVectorFromCenter);

      const canvasVectorFromCenterStart = vec2.create();
      vec2.scale(
        canvasVectorFromCenterStart,
        canvasUnitVectorFromCenter,
        canvasDiagonalLength * 0.05
      );

      const stPointLineCanvas1Start = vec2.create();
      const stPointLineCanvas2Start = vec2.create();
      vec2.add(
        stPointLineCanvas1Start,
        centerCanvas,
        canvasVectorFromCenterStart
      );
      vec2.subtract(
        stPointLineCanvas2Start,
        centerCanvas,
        canvasVectorFromCenterStart
      );

      const lineSegment1 = {
        start: {
          x: stPointLineCanvas1Start[0],
          y: stPointLineCanvas1Start[1],
        },
        end: {
          x: stPointLineCanvas1[0],
          y: stPointLineCanvas1[1],
        },
      };

      const distanceToPoint1 = lineSegment.distanceToPoint(
        [lineSegment1.start.x, lineSegment1.start.y],
        [lineSegment1.end.x, lineSegment1.end.y],
        [canvasCoords[0], canvasCoords[1]]
      );

      const lineSegment2 = {
        start: {
          x: stPointLineCanvas2Start[0],
          y: stPointLineCanvas2Start[1],
        },
        end: {
          x: stPointLineCanvas2[0],
          y: stPointLineCanvas2[1],
        },
      };

      const distanceToPoint2 = lineSegment.distanceToPoint(
        [lineSegment2.start.x, lineSegment2.start.y],
        [lineSegment2.end.x, lineSegment2.end.y],
        [canvasCoords[0], canvasCoords[1]]
      );

      if (distanceToPoint1 <= proximity || distanceToPoint2 <= proximity) {
        viewportIdArray.push(otherViewport.id); // we still need this to draw inactive slab thickness handles
        data.handles.activeOperation = null; // no operation
      }

      // slab thickness handles are in couples
      i++;
    }

    data.activeViewportIds = [...viewportIdArray];

    this.editData = {
      annotation,
    };

    return data.handles.activeOperation === OPERATION.DRAG ? true : false;
  }

  _registerViewportWithManager = (
    renderingEngineId: string,
    viewportId: string
  ): void => {
    const enabledElement = getEnabledElementByIds(viewportId, renderingEngineId);
    if (!enabledElement) {
      return;
    }

    const { FrameOfReferenceUID, viewport } = enabledElement;
    if (!FrameOfReferenceUID) {
      console.warn(
        `[CrosshairsTool] No FrameOfReferenceUID for viewport "${viewportId}". Skipping manager registration.`
      );
      return;
    }

    const manager = getWorldPointManager();
    const viewportType = viewport instanceof StackViewport ? 'stack' : 'volume';
    const modality = this._getViewportModality(viewport);

    manager.registerViewport(FrameOfReferenceUID, {
      viewportId,
      renderingEngineId,
      viewportType,
      modality,
      onWorldPointChanged: (point: Types.Point3, senderViewportId?: string) => {
        if (senderViewportId === viewportId) {
          return;
        }
        this._onManagerPointChanged(point, viewportId, renderingEngineId, senderViewportId);
      },
    });
  };

  _unregisterViewportFromManager = (
    renderingEngineId: string,
    viewportId: string
  ): void => {
    const enabledElement = getEnabledElementByIds(viewportId, renderingEngineId);
    const forUID = enabledElement?.FrameOfReferenceUID || this._getOwnFrameOfReferenceUID();
    if (forUID) {
      const manager = getWorldPointManager();
      manager.unregisterViewport(forUID, renderingEngineId, viewportId);
    }
  };

  _onManagerPointChanged = (
  point: Types.Point3,
  targetViewportId: string,
  targetRenderingEngineId: string,
  senderViewportId?: string
): void => {
  // Ignore updates that originated from this very viewport.
  if (senderViewportId && senderViewportId === targetViewportId) {
    return;
  }
  
  this._localToolCenter = [...point] as Types.Point3;
  if (this._isFinitePoint3(point) && !this._isNearZeroPoint3(point)) {
    this._lastValidToolCenter = [...point] as Types.Point3;
  }

  const enabledElement = getEnabledElementByIds(
    targetViewportId,
    targetRenderingEngineId
  );
  if (!enabledElement) {
    triggerAnnotationRenderForViewportIds([targetViewportId]);
    return;
  }
  const { viewport } = enabledElement;

  this._debugLog('_onManagerPointChanged', {
    targetViewportId,
    senderViewportId,
    point: this._formatPoint3(point),
    viewportType: viewport instanceof StackViewport ? 'stack' : 'volume',
  });

  const isCrosshairDragging =
    this.editData?.annotation?.data?.handles?.activeOperation ===
    OPERATION.DRAG;

  if (viewport instanceof StackViewport) {
    // StackViewports are excluded from direct camera shifts in _dragCallback,
    // so they must receive updates via the manager even during a drag (enables
    // live scroll across 2D planes). Feedback loop is guarded by
    // _pendingProgrammaticSliceIndex.
    this._syncStackSlices(
      [{ viewportId: targetViewportId, renderingEngineId: targetRenderingEngineId }],
      point
    );
  } else {
    // MPR (Volume) viewports are already shifted directly by the drag source
    // via _applyDeltaShiftToSelectedViewportCameras; applying the manager
    // update again here would move them at double speed, so ignore it during a
    // drag.
    if (isCrosshairDragging) {
      return;
    }
    this._scrollVolumeViewportToPoint(viewport, point);
  }
  
  triggerAnnotationRenderForViewportIds([targetViewportId]);
};

  /**
   * Scrolls a non-Stack viewport's camera along its view plane normal so the
   * reformat plane passes through `worldPoint`, without panning in-plane.
   * Wrapped in _ignoreFiredEvents so the emitted CAMERA_MODIFIED doesn't loop back.
   */
  _scrollVolumeViewportToPoint = (
    viewport,
    worldPoint: Types.Point3
  ): void => {
    const camera = getViewportICamera(viewport);
    const { focalPoint, position, viewPlaneNormal } = camera;
    if (!focalPoint || !position || !viewPlaneNormal) {
      return;
    }

    const scroll =
      (worldPoint[0] - focalPoint[0]) * viewPlaneNormal[0] +
      (worldPoint[1] - focalPoint[1]) * viewPlaneNormal[1] +
      (worldPoint[2] - focalPoint[2]) * viewPlaneNormal[2];

    if (Math.abs(scroll) < 1e-3) {
      return;
    }

    const newFocalPoint: Types.Point3 = [
      focalPoint[0] + scroll * viewPlaneNormal[0],
      focalPoint[1] + scroll * viewPlaneNormal[1],
      focalPoint[2] + scroll * viewPlaneNormal[2],
    ];
    const newPosition: Types.Point3 = [
      position[0] + scroll * viewPlaneNormal[0],
      position[1] + scroll * viewPlaneNormal[1],
      position[2] + scroll * viewPlaneNormal[2],
    ];

    const previousIgnoreFiredEvents = this._ignoreFiredEvents;
    this._ignoreFiredEvents = true;
    try {
      if (csUtils.isGenericViewport(viewport)) {
        jumpToFocalPoint(viewport, newFocalPoint);
      } else {
        viewport.setCamera({ focalPoint: newFocalPoint, position: newPosition });
      }
      viewport.render();
    } finally {
      this._ignoreFiredEvents = previousIgnoreFiredEvents;
    }

    this._debugLog('_scrollVolumeViewportToPoint', {
      viewportId: viewport.id,
      scroll: scroll.toFixed(4),
      newFocalPoint: this._formatPoint3(newFocalPoint),
    });
  };

  _registerAllViewportsWithManager = (): void => {
    const viewportsInfo = this._getViewportsInfo();
    viewportsInfo.forEach(({ viewportId, renderingEngineId }) => {
      this._registerViewportWithManager(renderingEngineId, viewportId);
    });
  };

  _unregisterAllViewportsFromManager = (): void => {
    const viewportsInfo = this._getViewportsInfo();
    viewportsInfo.forEach(({ viewportId, renderingEngineId }) => {
      this._unregisterViewportFromManager(renderingEngineId, viewportId);
    });
  };

  _toViewportKey = (renderingEngineId: string, viewportId: string): string => {
    return `${renderingEngineId}::${viewportId}`;
  };

  _getViewportModality = (viewport: any): string => {
    try {
      if (typeof csUtils.getViewportModality === 'function') {
        return csUtils.getViewportModality(viewport);
      }
    } catch {
    }
    if (viewport?.modality) {
      return viewport.modality;
    }
    return 'CT';
  };

  _isDebugEnabled = (): boolean => {
    return (
      !!this.configuration?.debug ||
      !!(globalThis as { __CROSSHAIRS_DEBUG__?: boolean }).__CROSSHAIRS_DEBUG__
    );
  };

  _debugLog = (message: string, payload?: unknown): void => {
    if (!this._isDebugEnabled()) {
      return;
    }

    if (payload !== undefined) {
      console.info(`[CrosshairsDebug:${this.toolGroupId}] ${message}`, payload);
    } else {
      console.info(`[CrosshairsDebug:${this.toolGroupId}] ${message}`);
    }
  };

  _formatPoint3 = (point: Types.Point3 | null | undefined): string | null => {
    if (!point || point.length !== 3) {
      return null;
    }

    return point.map((v) => Number(v).toFixed(3)).join(',');
  };

  _getViewportsDebugInfo = (
    viewportsInfo: { viewportId: string; renderingEngineId: string }[]
  ): Array<{
    viewportId: string;
    renderingEngineId: string;
    enabled: boolean;
    frameOfReferenceUID?: string;
    focalPoint?: string | null;
    viewPlaneNormal?: string | null;
  }> => {
    return viewportsInfo.map(({ viewportId, renderingEngineId }) => {
      const enabledElement = getEnabledElementByIds(
        viewportId,
        renderingEngineId
      );
      const viewport = enabledElement?.viewport;
      const camera = viewport ? getViewportICamera(viewport) : null;

      return {
        viewportId,
        renderingEngineId,
        enabled: !!enabledElement,
        frameOfReferenceUID: enabledElement?.FrameOfReferenceUID,
        focalPoint: camera?.focalPoint
          ? this._formatPoint3(camera.focalPoint as Types.Point3)
          : null,
        viewPlaneNormal: camera?.viewPlaneNormal
          ? this._formatPoint3(camera.viewPlaneNormal as Types.Point3)
          : null,
      };
    });
  };

  _isNearZeroPoint3 = (point: Types.Point3, epsilon = 1e-3): boolean => {
    return (
      Math.abs(point[0]) < epsilon &&
      Math.abs(point[1]) < epsilon &&
      Math.abs(point[2]) < epsilon
    );
  };

  _commitToolCenter = (
    point: Types.Point3,
    {
      senderViewportId,
      senderRenderingEngineId,
    }: {
      senderViewportId?: string;
      senderRenderingEngineId?: string;
    } = {}
  ): void => {
    this._localToolCenter = [...point] as Types.Point3;

    const forUID = this._getOwnFrameOfReferenceUID();
    if (forUID) {
      const manager = getWorldPointManager();
      if (!manager.isInitialized(forUID)) {
        manager.initializePoint(forUID, point);
      } else {
        manager.setPoint(forUID, point, senderViewportId, senderRenderingEngineId);
      }
    }

    if (this._isFinitePoint3(point) && !this._isNearZeroPoint3(point)) {
      this._lastValidToolCenter = [...point] as Types.Point3;
    }
  };

  _isFinitePoint3 = (point: Types.Point3): boolean => {
    if (!point || point.length !== 3) {
      return false;
    }

    return (
      Number.isFinite(point[0]) &&
      Number.isFinite(point[1]) &&
      Number.isFinite(point[2])
    );
  };

  _bindToolGroupViewportListeners = (): void => {
    if (!this._toolGroupViewportAddedListener) {
      this._toolGroupViewportAddedListener = ((evt: CustomEvent) => {
        if (evt.detail?.toolGroupId !== this.toolGroupId) {
          return;
        }

        const { viewportId, renderingEngineId } = evt.detail;
        if (viewportId && renderingEngineId) {
          this._registerViewportWithManager(renderingEngineId, viewportId);
        }

        this._syncVolumeListenersWithToolGroup();
        this._scheduleDebouncedRecompute();
      }) as EventListener;
      eventTarget.addEventListener(
        Events.TOOLGROUP_VIEWPORT_ADDED,
        this._toolGroupViewportAddedListener
      );
    }

    if (!this._toolGroupViewportRemovedListener) {
      this._toolGroupViewportRemovedListener = ((evt: CustomEvent) => {
        if (evt.detail?.toolGroupId !== this.toolGroupId) {
          return;
        }

        const { viewportId, renderingEngineId } = evt.detail;
        if (viewportId && renderingEngineId) {
          this._unregisterViewportFromManager(renderingEngineId, viewportId);
        }

        this._syncVolumeListenersWithToolGroup();

        // Re-initialize + re-render the remaining viewports; removing one
        // (e.g. 2D -> 3D switch) can leave siblings' state stale otherwise.
        this._computeToolCenter(this._getViewportsInfo());
      }) as EventListener;
      eventTarget.addEventListener(
        Events.TOOLGROUP_VIEWPORT_REMOVED,
        this._toolGroupViewportRemovedListener
      );
    }

    if (!this._elementEnabledListener) {
      // Safety net: TOOLGROUP_VIEWPORT_ADDED can fire before the element is
      // actually enabled, so the initial render finds no enabled element.
      // Re-trigger once the element becomes available.
      this._elementEnabledListener = ((evt: CustomEvent) => {
        const { viewportId, renderingEngineId } = evt.detail || {};
        if (!viewportId) {
          return;
        }

        const isOwnViewport = this._getViewportsInfo().some(
          (info) =>
            info.viewportId === viewportId &&
            (!renderingEngineId || info.renderingEngineId === renderingEngineId)
        );

        if (!isOwnViewport) {
          return;
        }

        triggerAnnotationRenderForViewportIds([viewportId]);
      }) as EventListener;

      eventTarget.addEventListener(
        Enums.Events.ELEMENT_ENABLED,
        this._elementEnabledListener
      );
    }

    if (!this._imageRenderedListener) {
      // Extra safety net: IMAGE_RENDERED fires on every actual paint, so
      // re-requesting our annotation render for our own viewports is a
      // reliable catch-all for viewports that had nothing to draw earlier.
      this._imageRenderedListener = ((evt: CustomEvent) => {
        const { viewportId, renderingEngineId } = evt.detail || {};
        if (!viewportId || !renderingEngineId) {
          return;
        }

        const isOwnViewport = this._getViewportsInfo().some(
          (info) =>
            info.viewportId === viewportId &&
            info.renderingEngineId === renderingEngineId
        );

        if (!isOwnViewport) {
          return;
        }

        // First real render for this viewport: re-run initializeViewport once,
        // since its annotation may have been created before the
        // FrameOfReferenceUID settled.
        const key = this._toViewportKey(renderingEngineId, viewportId);
        if (!this._seenStackViewports.has(key)) {
          this._seenStackViewports.add(key);
          this.initializeViewport({ viewportId, renderingEngineId });
        }

        triggerAnnotationRenderForViewportIds([viewportId]);
      }) as EventListener;

      eventTarget.addEventListener(
        Enums.Events.IMAGE_RENDERED,
        this._imageRenderedListener
      );
    }
  };

  _unbindToolGroupViewportListeners = (): void => {
    if (this._toolGroupViewportAddedListener) {
      eventTarget.removeEventListener(
        Events.TOOLGROUP_VIEWPORT_ADDED,
        this._toolGroupViewportAddedListener
      );
      this._toolGroupViewportAddedListener = null;
    }

    if (this._toolGroupViewportRemovedListener) {
      eventTarget.removeEventListener(
        Events.TOOLGROUP_VIEWPORT_REMOVED,
        this._toolGroupViewportRemovedListener
      );
      this._toolGroupViewportRemovedListener = null;
    }

    if (this._elementEnabledListener) {
      eventTarget.removeEventListener(
        Enums.Events.ELEMENT_ENABLED,
        this._elementEnabledListener
      );
      this._elementEnabledListener = null;
    }

    if (this._imageRenderedListener) {
      eventTarget.removeEventListener(
        Enums.Events.IMAGE_RENDERED,
        this._imageRenderedListener
      );
      this._imageRenderedListener = null;
    }
  };

  _syncVolumeListenersWithToolGroup = (): void => {
    const viewportsInfo = this._getViewportsInfo();
    const activeViewportKeys = new Set<string>();

    viewportsInfo.forEach((viewportInfo) => {
      const { viewportId, renderingEngineId } = viewportInfo;
      const viewportKey = this._toViewportKey(renderingEngineId, viewportId);
      activeViewportKeys.add(viewportKey);

      const enabledElement = getEnabledElementByIds(
        viewportId,
        renderingEngineId
      );
      const existingListenerInfo =
        this._volumeViewportNewVolumeListeners.get(viewportKey);

      if (!enabledElement) {
        if (existingListenerInfo) {
          existingListenerInfo.element.removeEventListener(
            Enums.Events.VOLUME_VIEWPORT_NEW_VOLUME,
            existingListenerInfo.handler
          );
          this._volumeViewportNewVolumeListeners.delete(viewportKey);
        }
        return;
      }

      const { viewport } = enabledElement;
      const { element } = viewport;

      if (existingListenerInfo && existingListenerInfo.element !== element) {
        existingListenerInfo.element.removeEventListener(
          Enums.Events.VOLUME_VIEWPORT_NEW_VOLUME,
          existingListenerInfo.handler
        );
        this._volumeViewportNewVolumeListeners.delete(viewportKey);
      }

      if (this._volumeViewportNewVolumeListeners.has(viewportKey)) {
        return;
      }

      const handler = ((evt: Event) => this._onNewVolume(evt)) as EventListener;
      element.addEventListener(
        Enums.Events.VOLUME_VIEWPORT_NEW_VOLUME,
        handler
      );

      this._volumeViewportNewVolumeListeners.set(viewportKey, {
        element,
        handler,
      });
    });

    Array.from(this._volumeViewportNewVolumeListeners.entries()).forEach(
      ([viewportKey, listenerInfo]) => {
        if (activeViewportKeys.has(viewportKey)) {
          return;
        }

        listenerInfo.element.removeEventListener(
          Enums.Events.VOLUME_VIEWPORT_NEW_VOLUME,
          listenerInfo.handler
        );
        this._volumeViewportNewVolumeListeners.delete(viewportKey);
      }
    );
  };

  _clearAllVolumeListenersAndViewportState = (): void => {
    this._volumeViewportNewVolumeListeners.forEach((listenerInfo) => {
      listenerInfo.element.removeEventListener(
        Enums.Events.VOLUME_VIEWPORT_NEW_VOLUME,
        listenerInfo.handler
      );
    });

    this._volumeViewportNewVolumeListeners.clear();
  };

  _updateToolCenterFromCameraDelta = (
    viewport: Types.IVolumeViewport,
    evt: Event
  ): void => {
    if (!this._isFinitePoint3(this.toolCenter) || this._isNearZeroPoint3(this.toolCenter)) {
      this._debugLog('_updateToolCenterFromCameraDelta:skip-uninitialized', {
        toolCenter: this._formatPoint3(this.toolCenter),
      });
      return;
    }

    const eventDetail = (evt as CustomEvent).detail;
    if (!eventDetail?.previousCamera || !eventDetail?.camera) {
      return;
    }

    const { previousCamera, camera } = eventDetail;
    const oldFocalPoint = previousCamera.focalPoint as Types.Point3;
    const newFocalPoint = camera.focalPoint as Types.Point3;
    const viewPlaneNormal = camera.viewPlaneNormal as Types.Point3;

    if (!oldFocalPoint || !newFocalPoint || !viewPlaneNormal) {
      return;
    }

    const delta: Types.Point3 = [
      newFocalPoint[0] - oldFocalPoint[0],
      newFocalPoint[1] - oldFocalPoint[1],
      newFocalPoint[2] - oldFocalPoint[2],
    ];

    const deltaLength = Math.sqrt(
      delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]
    );

    if (deltaLength < 1e-3) {
      return;
    }

    if (deltaLength > 50) {
      this._debugLog('_updateToolCenterFromCameraDelta:skip-large-delta', {
        deltaLength,
        delta: this._formatPoint3(delta),
      });
      return;
    }

    const dotProduct =
      delta[0] * viewPlaneNormal[0] +
      delta[1] * viewPlaneNormal[1] +
      delta[2] * viewPlaneNormal[2];

    if (Math.abs(dotProduct) < 1e-3) {
      return;
    }

    const projectedDelta: Types.Point3 = [
      dotProduct * viewPlaneNormal[0],
      dotProduct * viewPlaneNormal[1],
      dotProduct * viewPlaneNormal[2],
    ];

    const newToolCenter: Types.Point3 = [
      this.toolCenter[0] + projectedDelta[0],
      this.toolCenter[1] + projectedDelta[1],
      this.toolCenter[2] + projectedDelta[2],
    ];

    this._debugLog('_updateToolCenterFromCameraDelta', {
      oldFocalPoint: this._formatPoint3(oldFocalPoint),
      newFocalPoint: this._formatPoint3(newFocalPoint),
      delta: this._formatPoint3(delta),
      projectedDelta: this._formatPoint3(projectedDelta),
      oldToolCenter: this._formatPoint3(this.toolCenter),
      newToolCenter: this._formatPoint3(newToolCenter),
    });

    // Commit via _commitToolCenter (which calls the manager internally).
    // Passing the source viewport id prevents the manager from notifying it
    // back (ping-pong), and _ignoreFiredEvents suppresses the CAMERA_MODIFIED
    // cascade from the resulting setCamera calls (breaks recursion).
    const renderingEngineId =
      (viewport as { renderingEngineId?: string }).renderingEngineId ??
      this._getViewportsInfo().find((info) => info.viewportId === viewport.id)
        ?.renderingEngineId;

    const previousIgnoreFiredEvents = this._ignoreFiredEvents;
    this._ignoreFiredEvents = true;
    try {
      this._commitToolCenter(newToolCenter, {
        senderViewportId: viewport.id,
        senderRenderingEngineId: renderingEngineId,
      });
    } finally {
      this._ignoreFiredEvents = previousIgnoreFiredEvents;
    }
  };

  _recomputeToolCenterFromAbsoluteCameras = ({
    emitEvent = true,
    updateViewportCameras = false,
  }: {
    emitEvent?: boolean;
    updateViewportCameras?: boolean;
  } = {}): Types.Point3 | null => {
    const viewportsInfo = this._getViewportsInfo();
    const forUID = this._getOwnFrameOfReferenceUID();
    const manager = getWorldPointManager();

    const planes: CameraPlane[] = [];
    viewportsInfo.forEach((viewportInfo) => {
      const enabledElement = getEnabledElementByIds(
        viewportInfo.viewportId,
        viewportInfo.renderingEngineId
      );
      if (!enabledElement) {
        return;
      }
      const camera = getViewportICamera(enabledElement.viewport);
      const normal = [...camera.viewPlaneNormal] as Types.Point3;
      const point = [...camera.focalPoint] as Types.Point3;
      if (this._isFinitePoint3(normal) && this._isFinitePoint3(point) && !this._isNearZeroPoint3(point)) {
        planes.push({ normal, point });
      }
    });

    const referencePoint =
      this._isFinitePoint3(this.toolCenter) && !this._isNearZeroPoint3(this.toolCenter)
        ? ([...this.toolCenter] as Types.Point3)
        : this._isFinitePoint3(this._lastValidToolCenter as Types.Point3) &&
            !this._isNearZeroPoint3(this._lastValidToolCenter as Types.Point3)
          ? ([...(this._lastValidToolCenter as Types.Point3)] as Types.Point3)
          : null;

    let toolCenter = forUID
      ? manager.recomputeFromCameras(forUID, planes, referencePoint)
      : null;

    const currentCenterIsMeaningful =
      this._isFinitePoint3(this.toolCenter) &&
      !this._isNearZeroPoint3(this.toolCenter);
    const lastValidCenterIsMeaningful =
      this._isFinitePoint3(this._lastValidToolCenter as Types.Point3) &&
      !this._isNearZeroPoint3(this._lastValidToolCenter as Types.Point3);

    if (toolCenter && this._isNearZeroPoint3(toolCenter)) {
      this._debugLog(
        '_recomputeToolCenterFromAbsoluteCameras:calculated-near-zero',
        {
          calculatedToolCenter: this._formatPoint3(toolCenter),
          currentCenter: this._formatPoint3(this.toolCenter),
          lastValidToolCenter: this._formatPoint3(this._lastValidToolCenter),
        }
      );

      if (currentCenterIsMeaningful) {
        toolCenter = [...this.toolCenter] as Types.Point3;
      } else if (lastValidCenterIsMeaningful) {
        toolCenter = [
          ...(this._lastValidToolCenter as Types.Point3),
        ] as Types.Point3;
      }
    }

    if (!toolCenter && this._lastValidToolCenter) {
      toolCenter = [...this._lastValidToolCenter] as Types.Point3;
    }

    if (!toolCenter) {
      this._debugLog('_recomputeToolCenterFromAbsoluteCameras:skip:no-center');
      return null;
    }

    const hasChanged = !csUtils.isEqual(this.toolCenter, toolCenter, 1e-3);
    if (!hasChanged) {
      this._debugLog('_recomputeToolCenterFromAbsoluteCameras:no-change', {
        toolCenter: this._formatPoint3(toolCenter),
      });
      return toolCenter;
    }

    if (updateViewportCameras) {
      this._debugLog(
        '_recomputeToolCenterFromAbsoluteCameras:apply-with-camera-update',
        {
          toolCenter: this._formatPoint3(toolCenter),
          emitEvent,
        }
      );
      this.setToolCenter(toolCenter, !emitEvent);
    } else {
      this._debugLog(
        '_recomputeToolCenterFromAbsoluteCameras:apply-state-only',
        {
          toolCenter: this._formatPoint3(toolCenter),
          emitEvent,
        }
      );
      this._commitToolCenter(toolCenter);

      if (emitEvent) {
        const currentPoint = forUID ? manager.getPoint(forUID) : null;
        triggerEvent(eventTarget, Events.CROSSHAIR_TOOL_CENTER_CHANGED, {
          toolGroupId: this.toolGroupId,
          toolCenter: currentPoint ?? this.toolCenter,
          FrameOfReferenceUID: forUID,
        });
      }
    }

    return toolCenter;
  };
}

function getMinimalCrosshairConfig(configuration) {
  const minimal = configuration?.minimal;

  if (!minimal?.enabled) {
    return {
      enabled: false,
      lineLengthInPx: 0,
    };
  }

  const lineLengthInPx =
    typeof minimal.lineLengthInPx === 'number' &&
    Number.isFinite(minimal.lineLengthInPx)
      ? minimal.lineLengthInPx
      : 40;

  return {
    enabled: true,
    lineLengthInPx: Math.max(0, lineLengthInPx),
  };
}

function getSegmentMidpoint(
  start: Types.Point2 | vec2,
  end: Types.Point2 | vec2
): Types.Point2 {
  const midpoint = vec2.create();
  vec2.add(midpoint, start, end);
  vec2.scale(midpoint, midpoint, 0.5);

  return midpoint as Types.Point2;
}

CrosshairsTool.toolName = 'Crosshairs';
export default CrosshairsTool;
