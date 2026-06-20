use bevy::prelude::*;
use voxel_physics::{Accelerations, FreezePhysics, IsStatic, PhysicsSet, RigidBody};

/// Pulls dynamic physics bodies toward the center of the planet.
pub(crate) struct GravityPlugin;

#[derive(Resource, Debug, Clone, Copy)]
pub(crate) struct PlanetGravity {
    pub center: Vec3,
    pub acceleration: f32,
}

impl Default for PlanetGravity {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            acceleration: 150.0,
        }
    }
}

impl Plugin for GravityPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlanetGravity>()
            // voxel-physics defaults to frozen so tools can opt in. This game
            // wants gravity-driven bodies immediately.
            .insert_resource(FreezePhysics(false))
            .add_systems(
                FixedUpdate,
                apply_planet_gravity
                    .in_set(PhysicsSet::Apply)
                    .run_if(|freeze: Res<FreezePhysics>| !freeze.0),
            );
    }
}

fn apply_planet_gravity(
    gravity: Res<PlanetGravity>,
    mut accelerations: ResMut<Accelerations>,
    bodies: Query<(Entity, &Transform), (With<RigidBody>, Without<IsStatic>)>,
) {
    for (body, transform) in &bodies {
        let to_center = gravity.center - transform.translation;
        let distance_squared = to_center.length_squared();
        if distance_squared <= f32::EPSILON {
            continue;
        }

        accelerations
            .apply_central_acceleration(body, to_center.normalize() * gravity.acceleration);
    }
}
