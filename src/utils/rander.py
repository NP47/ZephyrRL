import math
import pygame


def rotate(x, y, angle):
    new_x = math.cos(angle) * x - math.sin(angle) * y
    new_y = math.sin(angle) * x + math.cos(angle) * y
    return new_x, new_y

def draw_boat(x, y,theta_boat, width, length, surface):
        
        points = [(0, -length / 2), (width / 2, 0), (0, length / 2), (-width / 2, 0)]
        points = [rotate(x_, y_, theta_boat) for x_, y_ in points]
        points = list([(x_ + x, y_ + y) for x_, y_ in points])
        pygame.draw.polygon(surface, (225, 237, 233), points) 

def draw_sail(x, y,  theta_sail, theta_boat, sail_width, sail_height, surface):
        # Sail base center (aligned with the middle of the boat)
        sail_base_x, sail_base_y = x, y

        # Define the four corners of the rectangle
        sail_points = [
            (-sail_width / 2, 0),  (sail_width / 2, 0),               # Bottom-right corner
            (sail_width / 2, -sail_height),    # Top-right corner
            (-sail_width / 2, -sail_width)    # Top-left corner
        ]

        # Rotate the sail based on self.sail_angle + self.angle
        rotated_sail_points = [
            rotate(x, y, theta_sail + theta_boat) for x, y in sail_points
        ]

        # Translate the rotated sail to the sail base position
        sail_points_translated = [
            (x + sail_base_x, y + sail_base_y) for x, y in rotated_sail_points
        ]

        # Draw the sail as a rectangle
        pygame.draw.polygon(surface, (255, 0, 255), sail_points_translated)  # White sail
        
        
def draw_wind_arrow(surf, theta_wind, wind_speed):
    arrow_x = surf.get_width() * 0.9
    arrow_y = surf.get_height() * 0.1



    
    # Arrow dimensions
    arrow_length = 48
    arrow_tip_x = arrow_x + arrow_length * math.cos(theta_wind)
    arrow_tip_y = arrow_y + arrow_length * math.sin(theta_wind)
    arrow_base_x = arrow_x
    arrow_base_y = arrow_y

    # Draw the arrow line
    pygame.draw.line(surf, (255, 0, 0), (arrow_base_x, arrow_base_y), (arrow_tip_x, arrow_tip_y), 3)

    # Arrowhead points
    arrowhead_length = 10
    arrowhead_angle = math.pi / 6  # 30 degrees
    left_arrowhead_x = arrow_tip_x - arrowhead_length * math.cos(theta_wind - arrowhead_angle)
    left_arrowhead_y = arrow_tip_y - arrowhead_length * math.sin(theta_wind - arrowhead_angle)
    right_arrowhead_x = arrow_tip_x - arrowhead_length * math.cos(theta_wind + arrowhead_angle)
    right_arrowhead_y = arrow_tip_y - arrowhead_length * math.sin(theta_wind + arrowhead_angle)

        # Draw the arrowhead
    pygame.draw.polygon(surf, (255, 0, 0), [(arrow_tip_x, arrow_tip_y),
                                            (left_arrowhead_x, left_arrowhead_y),
                                            (right_arrowhead_x, right_arrowhead_y)])

    # Display wind speed next to the arrow
    font = pygame.font.Font(None, 24)
    wind_speed_text = font.render(f"{wind_speed:.1f} m/s", True, (255, 255, 255))
    text_offset_x = -15  # Offset to position text near the arrow
    text_offset_y = 60
    surf.blit(wind_speed_text, (arrow_x + text_offset_x, arrow_y + text_offset_y))

    pygame.draw.circle(surf, (0, 0, 0), (arrow_x, arrow_y), 50, width=2)
