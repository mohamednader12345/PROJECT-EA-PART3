import pygame
import random
import threading
import numpy as np



import DE_traffic_signal as de
import pso_fixed as pso

    
WIDTH, HEIGHT = 1350, 800
CLR_GRASS = (34, 139, 34)
CLR_ROAD = (45, 45, 45)
CLR_PANEL = (25, 30, 40)
CLR_YELLOW_LINE = (255, 210, 0)

class Vehicle:
    def __init__(self, x, y, direction, lane):
        self.x, self.y = x, y
        self.direction = direction 
        self.lane = lane
        self.color = random.choice([(180, 0, 0), (0, 80, 180), (210, 210, 210), (190, 140, 0)])
        self.speed_max = random.uniform(1.8, 2.4)
        self.speed = self.speed_max
        self.w, self.h = (52, 26) if direction in ['LEFT', 'RIGHT'] else (26, 52)

    def draw(self, screen):
      
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.w, self.h), border_radius=4)
        
      
        glass_clr = (135, 206, 235)
        f_light = (255, 255, 190) 
        r_light = (200, 0, 0)     
        
       
        show_brake_lights = (self.speed == 0)

        if self.direction == 'RIGHT':
            pygame.draw.rect(screen, glass_clr, (self.x + 36, self.y + 2, 8, self.h - 4))
            pygame.draw.rect(screen, f_light, (self.x + self.w - 5, self.y + 2, 5, 5))
            pygame.draw.rect(screen, f_light, (self.x + self.w - 5, self.y + self.h - 7, 5, 5))
            if show_brake_lights:
                pygame.draw.rect(screen, r_light, (self.x, self.y + 2, 3, 5))
                pygame.draw.rect(screen, r_light, (self.x, self.y + self.h - 7, 3, 5))
            
        elif self.direction == 'LEFT':
            pygame.draw.rect(screen, glass_clr, (self.x + 8, self.y + 2, 8, self.h - 4))
            pygame.draw.rect(screen, f_light, (self.x, self.y + 2, 5, 5))
            pygame.draw.rect(screen, f_light, (self.x, self.y + self.h - 7, 5, 5))
            if show_brake_lights:
                pygame.draw.rect(screen, r_light, (self.x + self.w - 3, self.y + 2, 3, 5))
                pygame.draw.rect(screen, r_light, (self.x + self.w - 3, self.y + self.h - 7, 3, 5))
            
        elif self.direction == 'DOWN':
            pygame.draw.rect(screen, glass_clr, (self.x + 2, self.y + 36, self.w - 4, 8))
            pygame.draw.rect(screen, f_light, (self.x + 2, self.y + self.h - 5, 5, 5))
            pygame.draw.rect(screen, f_light, (self.x + self.w - 7, self.y + self.h - 5, 5, 5))
            if show_brake_lights:
                pygame.draw.rect(screen, r_light, (self.x + 2, self.y, 5, 3))
                pygame.draw.rect(screen, r_light, (self.x + self.w - 7, self.y, 5, 3))
            
        elif self.direction == 'UP':
            pygame.draw.rect(screen, glass_clr, (self.x + 2, self.y + 8, self.w - 4, 8))
            pygame.draw.rect(screen, f_light, (self.x + 2, self.y, 5, 5))
            pygame.draw.rect(screen, f_light, (self.x + self.w - 7, self.y, 5, 5))
            if show_brake_lights:
                pygame.draw.rect(screen, r_light, (self.x + 2, self.y + self.h - 3, 5, 3))
                pygame.draw.rect(screen, r_light, (self.x + self.w - 7, self.y + self.h - 3, 5, 3))

    def move(self, signal_state, stop_line, others):
        can_move = True
        safe_dist = 85 
        
        if self.direction == 'RIGHT': sensor_rect = pygame.Rect(self.x + self.w, self.y - 8, safe_dist, self.h + 16)
        elif self.direction == 'LEFT': sensor_rect = pygame.Rect(self.x - safe_dist, self.y - 8, safe_dist, self.h + 16)
        elif self.direction == 'DOWN': sensor_rect = pygame.Rect(self.x - 8, self.y + self.h, self.w + 16, safe_dist)
        else: sensor_rect = pygame.Rect(self.x - 8, self.y - safe_dist, self.w + 16, safe_dist)

        for v in others:
            if v == self: continue
            if sensor_rect.colliderect(pygame.Rect(v.x, v.y, v.w, v.h)):
                can_move = False
                break
        
        dist_stop = 9999
        is_past_stop = False 
        
        if self.direction == 'RIGHT': 
            dist_stop = stop_line - (self.x + self.w)
            if self.x > stop_line: is_past_stop = True
        elif self.direction == 'LEFT': 
            dist_stop = self.x - stop_line
            if self.x + self.w < stop_line: is_past_stop = True
        elif self.direction == 'DOWN': 
            dist_stop = stop_line - (self.y + self.h)
            if self.y > stop_line: is_past_stop = True
        elif self.direction == 'UP': 
            dist_stop = self.y - stop_line
            if self.y + self.h < stop_line: is_past_stop = True

        if not is_past_stop:
            if signal_state in ['RED', 'YELLOW'] and 0 < dist_stop < 65:
                can_move = False

        if can_move:
            self.speed = min(self.speed + 0.1, self.speed_max)
            if self.direction == 'RIGHT': self.x += self.speed
            elif self.direction == 'LEFT': self.x -= self.speed
            elif self.direction == 'DOWN': self.y += self.speed
            elif self.direction == 'UP': self.y -= self.speed
        else:
            self.speed = 0 


class ConfigButton:
    def __init__(self, x, y, label, attr, step, min_v, max_v):
        self.label, self.attr, self.step = label, attr, step
        self.min_v, self.max_v = min_v, max_v
        self.rect_up = pygame.Rect(x + 210, y, 35, 25)
        self.rect_dn = pygame.Rect(x + 210, y + 30, 35, 25)

    def draw(self, screen, font, obj):
        val = getattr(obj, self.attr)
        txt = font.render(f"{self.label}: {round(val, 2)}", True, (240, 240, 240))
        screen.blit(txt, (self.rect_up.x - 210, self.rect_up.y + 12))
        pygame.draw.rect(screen, (60, 65, 80), self.rect_up, border_radius=4)
        pygame.draw.rect(screen, (60, 65, 80), self.rect_dn, border_radius=4)
        screen.blit(font.render("+", True, (255,255,255)), (self.rect_up.x+12, self.rect_up.y))
        screen.blit(font.render("-", True, (255,255,255)), (self.rect_dn.x+14, self.rect_dn.y))

    def handle(self, pos, obj):
        v = getattr(obj, self.attr)
        if self.rect_up.collidepoint(pos): setattr(obj, self.attr, min(self.max_v, v + self.step))
        if self.rect_dn.collidepoint(pos): setattr(obj, self.attr, max(self.min_v, v - self.step))

class TrafficSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Segoe UI", 16, bold=True)
        self.pop_size, self.gens, self.mutation, self.num_runs = 30, 50, 0.5, 5
        self.selected_algo, self.is_optimizing = "DE", False 
        self.reset_simulation()
        self.controls = [
            ConfigButton(1000, 150, "Population", "pop_size", 5, 5, 100),
            ConfigButton(1000, 210, "Generations", "gens", 5, 2, 100),
            ConfigButton(1000, 270, "Mutation/F", "mutation", 0.05, 0.1, 1.0),
            ConfigButton(1000, 330, "Cycles/Runs", "num_runs", 1, 1, 20)
        ]
        self.btn_pso = pygame.Rect(1000, 80, 140, 40)
        self.btn_de = pygame.Rect(1160, 80, 140, 40)

    def reset_simulation(self):
        self.vehicles = []
        self.best_solution = [30.0, 30.0, 30.0, 30.0]
        self.times = [[30, 30], [30, 30]]
        self.timers = [30.0, 30.0]
        self.phases = ['NS', 'NS']

    def get_lane_queues(self):
        queues = [0, 0, 0, 0] 
        for v in self.vehicles:
            if v.speed < 0.2:
                if v.x < 450:
                    if v.direction in ['UP', 'DOWN']: queues[0] += 1
                    else: queues[1] += 1
                else:
                    if v.direction in ['UP', 'DOWN']: queues[2] += 1
                    else: queues[3] += 1
        return queues

    def start_optimization_thread(self):
        self.is_optimizing = True
        threading.Thread(target=self.run_external_optimization).start()

    def run_external_optimization(self):
        try:
            if self.selected_algo == "DE":
                de.POPULATION_SIZE, de.NUM_GENERATIONS, de.NUM_RUN, de.F = int(self.pop_size), int(self.gens), int(self.num_runs), self.mutation
                seeds = de.load_or_create_seeds(de.NUM_RUN)
                best_ind = None
                for s in seeds:
                    random.seed(s); np.random.seed(s)
                    pop = de.toolbox.populationCreator(n=de.POPULATION_SIZE)
                    traffic = de.generate_traffic_stream(de.SIM_HORIZON) 
                    for ind in pop: ind.fitness.values = de.toolbox.evaluate(ind, traffic)
                    for g in range(de.NUM_GENERATIONS):
                        for i in range(len(pop)):
                            a, b, c = de.selectedIndices(len(pop), i)
                            mutant = de.mutation(pop[a], pop[b], pop[c], de.F, de.MIN_GREEN, de.MAX_GREEN)
                            trial = de.crossOver(pop[i], mutant, de.CR)
                            trial.fitness.values = de.toolbox.evaluate(trial, traffic)
                            if trial.fitness.values[0] < pop[i].fitness.values[0]: pop[i] = trial
                            if best_ind is None or pop[i].fitness.values[0] < best_ind.fitness.values[0]:
                                best_ind = de.creator.Individual(pop[i]); best_ind.fitness.values = pop[i].fitness.values
                res = list(best_ind)
            else:
                
                pso.POPULATION_SIZE = int(self.pop_size)
                pso.NUM_GENERATIONS = int(self.gens)
                pso.NUM_RUNS = int(self.num_runs) 
                
                seeds = pso.load_or_create_seeds(pso.NUM_RUNS)
                global_best = None
                
                for s in seeds:
                    random.seed(s)
                    np.random.seed(s)
                    traffic = pso.generate_traffic_stream(pso.SIM_HORIZON)
                    pop = pso.toolbox.populationCreator(n=pso.POPULATION_SIZE)
                    
                    for g in range(pso.NUM_GENERATIONS):
                        for p in pop:
                            p.fitness.values = pso.toolbox.evaluate(p, traffic)
                           
                            if p.best is None or p.fitness.values[0] < p.best.fitness.values[0]:
                                p.best = pso.creator.Particle(p)
                                p.best.fitness.values = p.fitness.values
                            
                            if global_best is None or p.fitness.values[0] < global_best.fitness.values[0]:
                                global_best = pso.creator.Particle(p)
                                global_best.fitness.values = p.fitness.values
                        
                       
                        for p in pop:
                            pso.toolbox.update(p, global_best, pso.W_FIXED) 
                
                res = list(global_best)

            self.best_solution = [round(x, 1) for x in res]
            self.times = [[res[0], res[1]], [res[0], res[1]]]
            self.timers = [res[0], res[0]]
        except Exception as e:
            print(f"Error: {e}")
            
        self.is_optimizing = False

    def draw_dashed_line(self, screen, color, start_pos, end_pos, width=3, dash_len=15):
        x1, y1 = start_pos
        x2, y2 = end_pos
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        dashes = int(length / (2 * dash_len))
        for i in range(dashes):
            start = (x1 + (x2-x1) * i / dashes, y1 + (y2-y1) * i / dashes)
            end = (x1 + (x2-x1) * (i + 0.5) / dashes, y1 + (y2-y1) * (i + 0.5) / dashes)
            pygame.draw.line(screen, color, start, end, width)

    def draw_env(self):
        self.screen.fill(CLR_GRASS)
        pygame.draw.rect(self.screen, CLR_ROAD, (0, 350, 950, 160)) 
        pygame.draw.rect(self.screen, CLR_ROAD, (150, 0, 160, 800)) 
        pygame.draw.rect(self.screen, CLR_ROAD, (700, 0, 160, 800)) 
        self.draw_dashed_line(self.screen, CLR_YELLOW_LINE, (0, 428), (950, 428))
        self.draw_dashed_line(self.screen, CLR_YELLOW_LINE, (228, 0), (228, 800))
        self.draw_dashed_line(self.screen, CLR_YELLOW_LINE, (778, 0), (778, 800))

    def draw_traffic_light(self, x, y, state, timer):
        pygame.draw.rect(self.screen, (30, 30, 30), (x, y, 32, 110), border_radius=5)
        colors = {'RED': (255, 0, 0), 'YELLOW': (255, 255, 0), 'GREEN': (0, 255, 0)}
        for i, (k, v) in enumerate(colors.items()):
            c = v if state == k else (v[0]//4, v[1]//4, v[2]//4)
            pygame.draw.circle(self.screen, c, (x+16, y+20+(i*35)), 10)
        txt = self.font.render(str(int(timer)), True, (255, 255, 255))
        self.screen.blit(txt, (x+4, y-25))

    def run(self):
        while True:
            self.draw_env()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
                if event.type == pygame.MOUSEBUTTONDOWN and not self.is_optimizing:
                    for c in self.controls: c.handle(event.pos, self)
                    if self.btn_pso.collidepoint(event.pos): self.selected_algo = "PSO"
                    if self.btn_de.collidepoint(event.pos): self.selected_algo = "DE"
                    btn_run_rect = pygame.Rect(1000, 700, 300, 60)
                    if btn_run_rect.collidepoint(event.pos): self.start_optimization_thread()

            if not self.is_optimizing:
                for i in range(2):
                    self.timers[i] -= 1/60
                    if self.timers[i] <= 0:
                        self.phases[i] = 'EW' if self.phases[i] == 'NS' else 'NS'
                        self.timers[i] = self.times[i][0] if self.phases[i] == 'NS' else self.times[i][1]
                
                if random.random() < 0.02: 
                    d = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
                    if d == 'DOWN': self.vehicles.append(Vehicle(random.choice([158, 708]), -100, 'DOWN', 'L1'))
                    elif d == 'UP': self.vehicles.append(Vehicle(random.choice([272, 822]), 900, 'UP', 'L2'))
                    elif d == 'RIGHT': self.vehicles.append(Vehicle(-100, 438, 'RIGHT', 'L3'))
                    elif d == 'LEFT': self.vehicles.append(Vehicle(1100, 362, 'LEFT', 'L4'))
                
                for v in self.vehicles[:]:
                    idx = 0 if v.x < 450 else 1
                    stop = 350 if v.direction == 'DOWN' else (510 if v.direction == 'UP' else (150 if idx == 0 else 700))
                    if v.direction == 'LEFT': stop = 860 if idx == 1 else 310
                    state = ('GREEN' if self.timers[idx] > 4 else 'YELLOW') if (self.phases[idx] == 'NS') == (v.direction in ['UP', 'DOWN']) else 'RED'
                    v.move(state, stop, self.vehicles)
                    v.draw(self.screen)
                    if v.x > 1400 or v.x < -200 or v.y > 950 or v.y < -200: self.vehicles.remove(v)

            pygame.draw.rect(self.screen, CLR_PANEL, (950, 0, 400, HEIGHT))
            p_clr = (0, 180, 200) if self.selected_algo == "PSO" else (60, 60, 70)
            d_clr = (0, 180, 200) if self.selected_algo == "DE" else (60, 60, 70)
            pygame.draw.rect(self.screen, p_clr, self.btn_pso, border_radius=5)
            pygame.draw.rect(self.screen, d_clr, self.btn_de, border_radius=5)
            self.screen.blit(self.font.render("PSO", True, (255,255,255)), (1050, 88))
            self.screen.blit(self.font.render("DE", True, (255,255,255)), (1215, 88))
            for c in self.controls: c.draw(self.screen, self.font, self)
            
            qs = self.get_lane_queues()
            self.screen.blit(self.font.render(f"Real-time Queues: {qs}", True, (255, 255, 0)), (1015, 385))

            self.screen.blit(self.font.render("SYSTEM READY" if not self.is_optimizing else "OPTIMIZING...", True, (0, 255, 180)), (1015, 415))
            labels = ["NS1 Time", "EW1 Time", "NS2 Time", "EW2 Time"]
            for i in range(4): self.screen.blit(self.font.render(f"{labels[i]}: {self.best_solution[i % 2]}s", True, (200, 200, 200)), (1025, 460+(i*35)))
            
            btn_run_rect = pygame.Rect(1000, 700, 300, 60)
            pygame.draw.rect(self.screen, (255, 160, 0), btn_run_rect, border_radius=8)
            self.screen.blit(self.font.render(f"RUN {self.selected_algo} ENGINE" if not self.is_optimizing else "PLEASE WAIT...", True, (0,0,0)), (1030, 720))
            
            for i, px in enumerate([320, 870]):
                s_ns = 'GREEN' if self.timers[i] > 4 and self.phases[i] == 'NS' else ('YELLOW' if self.timers[i] <= 4 and self.phases[i] == 'NS' else 'RED')
                s_ew = 'GREEN' if self.timers[i] > 4 and self.phases[i] == 'EW' else ('YELLOW' if self.timers[i] <= 4 and self.phases[i] == 'EW' else 'RED')
                self.draw_traffic_light(px, 240, s_ns, self.timers[i])
                self.draw_traffic_light(px-210, 520, s_ew, self.timers[i])
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    TrafficSimulation().run()