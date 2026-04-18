import { Object3DNode } from "@react-three/fiber"
import {
  Mesh,
  Group,
  AmbientLight,
  DirectionalLight,
  SphereGeometry,
  CylinderGeometry,
  MeshStandardMaterial,
  Object3D,
} from "three"

declare global {
  namespace JSX {
    interface IntrinsicElements {
      group: Object3DNode<Group, typeof Group>
      mesh: Object3DNode<Mesh, typeof Mesh>
      ambientLight: Object3DNode<AmbientLight, typeof AmbientLight>
      directionalLight: Object3DNode<DirectionalLight, typeof DirectionalLight>
      sphereGeometry: Object3DNode<SphereGeometry, typeof SphereGeometry>
      cylinderGeometry: Object3DNode<CylinderGeometry, typeof CylinderGeometry>
      meshStandardMaterial: Object3DNode<MeshStandardMaterial, typeof MeshStandardMaterial>
      primitive: { object?: Object3D; [key: string]: unknown }
    }
  }
}
