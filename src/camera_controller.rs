use bevy::prelude::*;
use bevy::render::view::{Hdr, Msaa};

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Hdr,
        Msaa::Off,
        Transform::from_xyz(0.0, 4200.0, 120.0),
    ));
}
