extends RayCast2D

@onready var beam: Line2D = $Line2D

@export var max_distance: float = 800.0
@export var build_time: float = 0.15
@export var life_time: float = 0.2

var firing := false
var progress := 0.0
var life_timer := 0.0

var beam_particles: GPUParticles2D
var hit_particles: GPUParticles2D


func _ready():
	_setup_particles()

	beam.visible = false
	beam_particles.emitting = false
	hit_particles.emitting = false


# ---------------------------
# INPUT
# ---------------------------
func _physics_process(delta):
	if Input.is_key_pressed(KEY_TAB) and not firing:
		start_laser()

	if firing:
		update_laser(delta)


# ---------------------------
# LASER START
# ---------------------------
func start_laser():
	firing = true
	progress = 0.0
	life_timer = life_time

	beam.visible = true
	beam_particles.emitting = true


# ---------------------------
# UPDATE
# ---------------------------
func update_laser(delta):
	if progress < 1.0:
		progress += delta / build_time
		progress = min(progress, 1.0)

	var direction = Vector2.RIGHT
	var current_length = max_distance * progress

	target_position = direction * current_length
	force_raycast_update()

	var end_point = direction * current_length

	# HIT
	if is_colliding():
		var collision_point = get_collision_point()
		end_point = to_local(collision_point)

		hit_particles.global_position = collision_point
		hit_particles.restart()
		hit_particles.emitting = true
	else:
		hit_particles.emitting = false

	# LINE
	beam.clear_points()
	beam.add_point(Vector2.ZERO)
	beam.add_point(end_point)

	# BEAM PARTICLES POSITION
	beam_particles.position = end_point * 0.5

	# AUTO STOP
	if progress >= 1.0:
		life_timer -= delta
		if life_timer <= 0:
			stop_laser()


# ---------------------------
# STOP
# ---------------------------
func stop_laser():
	firing = false

	beam.visible = false
	beam.clear_points()

	beam_particles.emitting = false
	hit_particles.emitting = false


# ---------------------------
# PARTICLES SETUP (ALL IN CODE)
# ---------------------------
func _setup_particles():
	# ===== BEAM PARTICLES =====
	beam_particles = GPUParticles2D.new()
	add_child(beam_particles)

	var beam_mat = ParticleProcessMaterial.new()
	beam_mat.direction = Vector3(1, 0, 0)
	beam_mat.spread = 10.0
	beam_mat.initial_velocity_min = 80.0
	beam_mat.initial_velocity_max = 200.0
	beam_mat.scale_min = 0.2
	beam_mat.scale_max = 0.6

	var beam_color = Gradient.new()
	beam_color.colors = PackedColorArray([
		Color(0, 1, 1, 1),
		Color(1, 1, 1, 1),
		Color(0, 1, 1, 0)
	])
	beam_mat.color_ramp = beam_color

	beam_particles.process_material = beam_mat
	beam_particles.amount = 40
	beam_particles.lifetime = 0.15
	beam_particles.one_shot = false
	beam_particles.emitting = false
	beam_particles.local_coords = true


	# ===== HIT PARTICLES =====
	hit_particles = GPUParticles2D.new()
	add_child(hit_particles)

	var hit_mat = ParticleProcessMaterial.new()
	hit_mat.direction = Vector3(0, 0, 0)
	hit_mat.spread = 180.0
	hit_mat.initial_velocity_min = 200.0
	hit_mat.initial_velocity_max = 600.0
	hit_mat.scale_min = 0.3
	hit_mat.scale_max = 1.0

	var hit_color = Gradient.new()
	hit_color.colors = PackedColorArray([
		Color(1, 1, 1, 1),
		Color(1, 0.5, 0, 1),
		Color(1, 0.5, 0, 0)
	])
	hit_mat.color_ramp = hit_color

	hit_particles.process_material = hit_mat
	hit_particles.amount = 30
	hit_particles.lifetime = 0.3
	hit_particles.one_shot = true
	hit_particles.emitting = false
