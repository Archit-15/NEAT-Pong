import pygame
from pong import Game
import neat
import os
import pickle

WIDTH , HEIGHT = 700,500
FPS = 60
WINNING_SCORE = 5
WHITE = (255,255,255)
SCORE_FONT =pygame.font.SysFont("comicsans", 50)

class PongGame:
    def __init__(self,window,width,height):
        self.game = Game(window,WIDTH,HEIGHT)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball
    
    def test_ai(self,genome,config):
        net = neat.nn.FeedForwardNetwork.create(genome,config)

        window = pygame.display.set_mode((WIDTH,HEIGHT))

        CLOCK = pygame.time.Clock()

        run = True
        while run:
            CLOCK.tick(FPS)
            self.game.draw(False,True)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()

            if(keys[pygame.K_w]):
                self.game.move_paddle(left=True,up=True)
            elif(keys[pygame.K_s]):
                self.game.move_paddle(left=True,up=False)

            output = net.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x-self.ball.x)))
            decision = output.index(max(output))
            if decision==0:
                pass
            elif decision==1:
                self.game.move_paddle(left=False,up=True)
            else:
                self.game.move_paddle(left=False,up = False)

            game_Info = self.game.loop()

            won = False
            
            if(self.game.left_score>=WINNING_SCORE):
                won = True
                wintext = "Left Player WON"
            elif(self.game.right_score>=WINNING_SCORE):
                won = True
                wintext = "Right Player WON"
            
            if won:
                text = SCORE_FONT.render(wintext,1,WHITE)
                window.blit(text,(WIDTH//2-text.get_width()//2,HEIGHT//2-text.get_height()//2))
                pygame.display.update()
                pygame.time.delay(5000)
                self.game.reset()
        
            self.game.draw(True,False)
            pygame.display.update()

        pygame.quit()

    def train_ai(self,genome1,genome2,config):
        #This will create a nueral network using genome1 and the configuration provided 
        #genome1 is the representation of nueral netwrok's genetic encoding(structure and conn. weights)
        net1 = neat.nn.FeedForwardNetwork.create(genome1,config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2,config)
        
        run = True
        while run :
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
        
            output1 = net1.activate((self.left_paddle.y,self.ball.y,abs(self.left_paddle.x-self.ball.x)))
            decision1 = output1.index(max(output1))
            if decision1==0:
                pass
            elif decision1==1:
                self.game.move_paddle(left=True,up=True)
            else:
                self.game.move_paddle(left=True,up=False)

            output2 = net2.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x-self.ball.x)))
            decision2 = output2.index(max(output2))
            if decision2==0:
                pass
            elif decision2==1:
                self.game.move_paddle(left=False,up=True)
            else:
                self.game.move_paddle(left=False,up = False)


            game_info = self.game.loop()
            self.game.draw(draw_hits=True,draw_score=False)
            pygame.display.update()

    #We are ending the loop , once even one side misses because if we keep on running the loop for a lot 
    #of turns then it will take a lot of titme and the fitness score will increase unfairly. So we only
    #do it once as it is more time effecient.
            if game_info.left_score >= 1 or game_info.right_score>=1 or game_info.left_hits>50:
                self.calculate_fitness(genome1,genome2,game_info)
                break

    def calculate_fitness(self,genome1,genome2,game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits


#genomes is a list of tuples where each tuple has genome id and the genome object
def eval_genomes(genomes,config):
    width , height = 700,500
    window = pygame.display.set_mode((width,height))

    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes)-1:
            break
        genome1.fitness = 0
        for genome_id2,genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness ==None else genome2.fitness
            game = PongGame(window,width,height)
            game.train_ai(genome1,genome2,config)


def run_neat(config):
    p = neat.Population(config)
    #StdOutReporter will output the information about the evolution processs to console in real time
    #and add_reporter is adding that reproter to our population object
    p.add_reporter(neat.StdOutReporter(True))

    #This is a different reporter it will show info like mean fittness and other relevant statistcs
    #after the evolution is over
    stats = neat.StatisticsReporter()   
    p.add_reporter(stats)   

    #This will create checkpoints after every generation, it will create a new file which will save
    #the state of entire population(weight,current generation number etc.).Using this we can again 
    #begin evolution from any specific gen.
    p.add_reporter(neat.Checkpointer(1))
    # We can use this "p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-27')" to start 
    # from an exsiting checkpoint

    #This method starts the nueroevolution process and takes the fitnesss function and the number of
    #generations for which the process is going to run. It calculates the fitness and does its mutations
    #and goes to next gen
    winner = p.run(eval_genomes,50)
    #Either the best genome after 50 gen will be retunred or if a genome hits 400 fittness before 50
    #gen, then that will be returned
    with open("best1.pickle","wb") as f:
        pickle.dump(winner,f)

def test_ai(config):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    with open("best1.pickle", "rb") as f:
        winner = pickle.load(f)

        game = PongGame(window, width, height)
        game.test_ai(winner, config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")

    #Creating a config object which contains various settings and parameters for the neat algo
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)
    #run_neat(config)
    test_ai(config)