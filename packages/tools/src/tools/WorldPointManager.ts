import { vec3 } from 'gl-matrix';
import { utilities as csUtils, type Types } from '@cornerstonejs/core';

export type ViewportType = 'stack' | 'volume' | 'volume3d';

export interface WorldPointSubscriber {
  viewportId: string;
  renderingEngineId: string;
  viewportType: ViewportType;
  modality: string;
  onWorldPointChanged: (point: Types.Point3, senderViewportId?: string) => void;
}

export interface WorldPointState {
  point: Types.Point3;
  isInitialized: boolean;
  bounds: Types.Point3[] | null;
}

export interface WorldPointManagerConfig {
  supportedModalities: string[];
}

export interface CameraPlane {
  normal: Types.Point3;
  point: Types.Point3;
}

const NO_FOR_KEY = '__no_for__';

const DEFAULT_CONFIG: WorldPointManagerConfig = {
  supportedModalities: ['CT', 'MR', 'PT', 'US', 'XA', 'DX'],
};

class WorldPointManager {
  private _stateByFOR = new Map<string, WorldPointState>();
  private _subscribersByFOR = new Map<string, Map<string, WorldPointSubscriber>>();
  private _config: WorldPointManagerConfig;

  constructor(config?: Partial<WorldPointManagerConfig>) {
    this._config = {
      ...DEFAULT_CONFIG,
      ...config,
    };
  }

  get config(): WorldPointManagerConfig {
    return this._config;
  }

  setSupportedModalities(modalities: string[]): void {
    this._config.supportedModalities = [...modalities];
  }

  isModalitySupported(modality: string): boolean {
    return this._config.supportedModalities.includes(modality);
  }

  registerViewport(
    frameOfReferenceUID: string | undefined,
    subscriber: WorldPointSubscriber
  ): boolean {
    if (!this.isModalitySupported(subscriber.modality)) {
      console.warn(
        `[WorldPointManager] Modality "${subscriber.modality}" is not in the supported list. ` +
          `Viewport "${subscriber.viewportId}" will not participate in synchronization.`
      );
      return false;
    }

    const forKey = frameOfReferenceUID || NO_FOR_KEY;

    if (forKey === NO_FOR_KEY) {
      console.warn(
        `[WorldPointManager] FrameOfReferenceUID is missing for viewport "${subscriber.viewportId}". ` +
          `Registering in "${NO_FOR_KEY}" group — no synchronization will occur.`
      );
    }

    if (!this._subscribersByFOR.has(forKey)) {
      this._subscribersByFOR.set(forKey, new Map());
    }

    const subscribers = this._subscribersByFOR.get(forKey)!;
    const key = this._subscriberKey(subscriber.renderingEngineId, subscriber.viewportId);
    subscribers.set(key, subscriber);

    if (!this._stateByFOR.has(forKey)) {
      this._stateByFOR.set(forKey, {
        point: [0, 0, 0],
        isInitialized: false,
        bounds: null,
      });
    }

    const state = this._stateByFOR.get(forKey)!;
    if (state.isInitialized) {
      subscriber.onWorldPointChanged([...state.point] as Types.Point3);
    }

    return true;
  }

  unregisterViewport(
    frameOfReferenceUID: string | undefined,
    renderingEngineId: string,
    viewportId: string
  ): void {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    const subscribers = this._subscribersByFOR.get(forKey);
    if (!subscribers) {
      return;
    }

    const key = this._subscriberKey(renderingEngineId, viewportId);
    subscribers.delete(key);
  }

  initializePoint(
    frameOfReferenceUID: string | undefined,
    point: Types.Point3,
    bounds?: Types.Point3[] | null
  ): void {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;

    if (forKey === NO_FOR_KEY) {
      return;
    }

    let state = this._stateByFOR.get(forKey);
    if (!state) {
      state = { point: [0, 0, 0], isInitialized: false, bounds: null };
      this._stateByFOR.set(forKey, state);
    }

    if (state.isInitialized) {
      return;
    }

    const clamped = this._clampToBounds(point, bounds ?? state.bounds);
    state.point = [...clamped] as Types.Point3;
    state.isInitialized = true;
    if (bounds) {
      state.bounds = bounds;
    }

    this._notifySubscribers(forKey, state.point);
  }

  setPoint(
    frameOfReferenceUID: string | undefined,
    point: Types.Point3,
    senderViewportId?: string,
    senderRenderingEngineId?: string
  ): void {
    this.updatePoint(frameOfReferenceUID, point, senderViewportId, senderRenderingEngineId);
  }

  updatePoint(
    frameOfReferenceUID: string | undefined,
    point: Types.Point3,
    senderViewportId?: string,
    senderRenderingEngineId?: string
  ): void {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;

    if (forKey === NO_FOR_KEY) {
      return;
    }

    let state = this._stateByFOR.get(forKey);
    if (!state) {
      state = { point: [0, 0, 0], isInitialized: false, bounds: null };
      this._stateByFOR.set(forKey, state);
    }

    const clamped = this._clampToBounds(point, state.bounds);
    state.point = [...clamped] as Types.Point3;
    state.isInitialized = true;

    this._notifySubscribers(forKey, state.point, senderViewportId, senderRenderingEngineId);
  }

  getPoint(frameOfReferenceUID: string | undefined): Types.Point3 | null {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    const state = this._stateByFOR.get(forKey);
    if (!state || !state.isInitialized) {
      return null;
    }
    return [...state.point] as Types.Point3;
  }

  isInitialized(frameOfReferenceUID: string | undefined): boolean {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    const state = this._stateByFOR.get(forKey);
    return state?.isInitialized ?? false;
  }

  setBounds(
    frameOfReferenceUID: string | undefined,
    bounds: Types.Point3[] | null
  ): void {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    const state = this._stateByFOR.get(forKey);
    if (state) {
      state.bounds = bounds;
    }
  }

  getSubscriberCount(frameOfReferenceUID: string | undefined): number {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    const subscribers = this._subscribersByFOR.get(forKey);
    return subscribers?.size ?? 0;
  }

  getSubscribers(frameOfReferenceUID: string | undefined): WorldPointSubscriber[] {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    const subscribers = this._subscribersByFOR.get(forKey);
    if (!subscribers) {
      return [];
    }
    return Array.from(subscribers.values());
  }

  getViewportTypes(frameOfReferenceUID: string | undefined): ViewportType[] {
    const subscribers = this.getSubscribers(frameOfReferenceUID);
    return subscribers.map((s) => s.viewportType);
  }

  hasMultipleVolumeViewportsWithDifferentNormals(
    frameOfReferenceUID: string | undefined,
    planes: CameraPlane[]
  ): boolean {
    if (planes.length < 2) {
      return false;
    }
    const uniqueNormals: Types.Point3[] = [];
    for (const plane of planes) {
      const normal = [...plane.normal] as Types.Point3;
      vec3.normalize(normal, normal);
      const isDuplicate = uniqueNormals.some(
        (existing) =>
          csUtils.isEqual(existing, normal, 1e-3) ||
          csUtils.isOpposite(existing, normal, 1e-3)
      );
      if (!isDuplicate) {
        uniqueNormals.push(normal);
      }
    }
    return uniqueNormals.length >= 2;
  }

  recomputeFromCameras(
    frameOfReferenceUID: string | undefined,
    planes: CameraPlane[],
    referencePoint?: Types.Point3 | null
  ): Types.Point3 | null {
    if (planes.length < 2) {
      return null;
    }

    const uniquePlanes: CameraPlane[] = [];
    for (const plane of planes) {
      const normal = [...plane.normal] as Types.Point3;
      const point = [...plane.point] as Types.Point3;

      if (!this._isFinitePoint3(normal) || !this._isFinitePoint3(point)) {
        continue;
      }

      if (this._isNearZeroPoint3(point)) {
        continue;
      }

      vec3.normalize(normal, normal);

      const isDuplicate = uniquePlanes.some(
        (existing) =>
          csUtils.isEqual(existing.normal, normal, 1e-3) ||
          csUtils.isOpposite(existing.normal, normal, 1e-3)
      );

      if (!isDuplicate) {
        uniquePlanes.push({ normal, point });
      }
    }

    if (uniquePlanes.length < 2) {
      return null;
    }

    const firstPlane = csUtils.planar.planeEquation(
      uniquePlanes[0].normal,
      uniquePlanes[0].point
    );
    const secondPlane = csUtils.planar.planeEquation(
      uniquePlanes[1].normal,
      uniquePlanes[1].point
    );

    let thirdPlane;
    if (uniquePlanes.length >= 3) {
      thirdPlane = csUtils.planar.planeEquation(
        uniquePlanes[2].normal,
        uniquePlanes[2].point
      );
    } else {
      const thirdNormal = vec3.create() as Types.Point3;
      vec3.cross(thirdNormal, uniquePlanes[0].normal, uniquePlanes[1].normal);

      if (vec3.length(thirdNormal) < 1e-6) {
        return null;
      }

      vec3.normalize(thirdNormal, thirdNormal);

      const thirdPoint = referencePoint && this._isFinitePoint3(referencePoint)
        ? ([...referencePoint] as Types.Point3)
        : ([
            (uniquePlanes[0].point[0] + uniquePlanes[1].point[0]) * 0.5,
            (uniquePlanes[0].point[1] + uniquePlanes[1].point[1]) * 0.5,
            (uniquePlanes[0].point[2] + uniquePlanes[1].point[2]) * 0.5,
          ] as Types.Point3);

      thirdPlane = csUtils.planar.planeEquation(thirdNormal, thirdPoint);
    }

    const center = csUtils.planar.threePlaneIntersection(
      firstPlane,
      secondPlane,
      thirdPlane
    ) as Types.Point3;

    if (!this._isFinitePoint3(center)) {
      return null;
    }

    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    const state = this._stateByFOR.get(forKey);
    const clamped = this._clampToBounds(center, state?.bounds ?? null);

    this.updatePoint(frameOfReferenceUID, clamped);
    return [...clamped] as Types.Point3;
  }

  destroy(): void {
    this._stateByFOR.clear();
    this._subscribersByFOR.clear();
  }

  destroyFOR(frameOfReferenceUID: string | undefined): void {
    const forKey = frameOfReferenceUID || NO_FOR_KEY;
    this._stateByFOR.delete(forKey);
    this._subscribersByFOR.delete(forKey);
  }

  private _clampToBounds(
    point: Types.Point3,
    bounds: Types.Point3[] | null
  ): Types.Point3 {
    if (!bounds || bounds.length < 2) {
      return [...point] as Types.Point3;
    }

    const [min, max] = bounds;
    return [
      Math.max(min[0], Math.min(max[0], point[0])),
      Math.max(min[1], Math.min(max[1], point[1])),
      Math.max(min[2], Math.min(max[2], point[2])),
    ] as Types.Point3;
  }

  private _isFinitePoint3(point: Types.Point3): boolean {
    if (!point || point.length !== 3) {
      return false;
    }
    return (
      Number.isFinite(point[0]) &&
      Number.isFinite(point[1]) &&
      Number.isFinite(point[2])
    );
  }

  private _isNearZeroPoint3(point: Types.Point3, epsilon = 1e-3): boolean {
    return (
      Math.abs(point[0]) < epsilon &&
      Math.abs(point[1]) < epsilon &&
      Math.abs(point[2]) < epsilon
    );
  }

  private _notifySubscribers(
    forKey: string,
    point: Types.Point3,
    senderViewportId?: string,
    senderRenderingEngineId?: string
  ): void {
    const subscribers = this._subscribersByFOR.get(forKey);
    if (!subscribers) {
      return;
    }

    const senderKey = senderViewportId && senderRenderingEngineId
      ? this._subscriberKey(senderRenderingEngineId, senderViewportId)
      : null;

    for (const [key, subscriber] of subscribers) {
      if (key === senderKey) {
        continue;
      }
      try {
        subscriber.onWorldPointChanged([...point] as Types.Point3, senderViewportId);
      } catch (e) {
        console.error(
          `[WorldPointManager] Error in subscriber "${subscriber.viewportId}":`,
          e
        );
      }
    }
  }

  private _subscriberKey(renderingEngineId: string, viewportId: string): string {
    return `${renderingEngineId}::${viewportId}`;
  }
}

let _instance: WorldPointManager | null = null;

export function getWorldPointManager(config?: Partial<WorldPointManagerConfig>): WorldPointManager {
  if (!_instance) {
    _instance = new WorldPointManager(config);
  }
  return _instance;
}

export function resetWorldPointManager(): void {
  if (_instance) {
    _instance.destroy();
    _instance = null;
  }
}

export default WorldPointManager;
